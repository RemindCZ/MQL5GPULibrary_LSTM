#ifndef MQL5_GPU_TOPOLOGY_MQH
#define MQL5_GPU_TOPOLOGY_MQH
#include <MQL5GPULibrary.mqh>

string GPUErrorText()
{
   short buffer[1024];
   DN_GetError(buffer,1024);
   string text="";
   for(int i=0;i<1024 && buffer[i]!=0;i++) text+=ShortToString(buffer[i]);
   return text;
}

// Nodes separated by ';': type:width:activation:dropout:merge:parent,parent
// Parent -1 means external input. Nodes must reference earlier zero-based IDs.
bool GPUParseInteger(string text,int &value)
{
   StringTrimLeft(text); StringTrimRight(text);
   int size=StringLen(text),start=0;
   if(size==0 || size>11) return false;
   if(StringGetCharacter(text,0)=='-') start=1;
   if(start==size) return false;
   for(int i=start;i<size;i++) if(StringGetCharacter(text,i)<'0' || StringGetCharacter(text,i)>'9') return false;
   long number=StringToInteger(text);
   if(number<-2147483647 || number>2147483647) return false;
   value=(int)number;
   return true;
}

bool GPUBuildTopology(const int handle,const int features,const int sequence,
                      const int outputs,const string topology,string &error)
{
   if(!DN_BeginGraph(handle,features,sequence)) { error=GPUErrorText(); return false; }
   string nodes[];
   int count=StringSplit(topology,';',nodes);
   if(count<1 || count>1024) { error="Topology must contain 1..1024 nodes"; return false; }
   int last=-1;
   for(int n=0;n<count;n++)
   {
      string fields[];
      if(StringSplit(nodes[n],':',fields)!=6) { error="Each node needs six colon-separated fields"; return false; }
      for(int f=0;f<6;f++) { StringTrimLeft(fields[f]); StringTrimRight(fields[f]); }
      int type=-1,activation=-1,merge=-1;
      if(fields[0]=="lstm") type=0;
      if(fields[0]=="gru") type=1;
      if(fields[0]=="rnn") type=2;
      if(fields[0]=="dense") type=3;
      if(fields[0]=="identity") type=4;
      if(fields[2]=="linear") activation=0;
      if(fields[2]=="tanh") activation=1;
      if(fields[2]=="relu") activation=2;
      if(fields[2]=="sigmoid") activation=3;
      if(fields[4]=="concat") merge=0;
      if(fields[4]=="sum") merge=1;
      string tokens[];
      int parents[];
      int np=StringSplit(fields[5],',',tokens);
      if(np<1 || np>64) { error="Each node needs 1..64 parents"; return false; }
      ArrayResize(parents,np);
      int width;
      if(!GPUParseInteger(fields[1],width)) { error="Invalid integer width"; return false; }
      for(int j=0;j<np;j++) if(!GPUParseInteger(tokens[j],parents[j])) { error="Invalid parent ID"; return false; }
      // Dropout syntax is a decimal fraction (scientific notation is not used in this DSL).
      int dots=0,digits=0;
      for(int j=0;j<StringLen(fields[3]);j++)
      {
         ushort c=StringGetCharacter(fields[3],j);
         if(c=='.') dots++;
         else if(c>='0' && c<='9') digits++;
         else { error="Invalid decimal dropout"; return false; }
      }
      if(dots>1 || digits==0) { error="Invalid dropout"; return false; }
      last=DN_AddGraphNode(handle,type,width,activation,StringToDouble(fields[3]),parents,np,merge);
      if(last<0) { error="Node "+IntegerToString(n)+": "+GPUErrorText(); return false; }
   }
   if(!DN_SetGraphOutput(handle,last,outputs)) { error=GPUErrorText(); return false; }
   return true;
}
#endif
