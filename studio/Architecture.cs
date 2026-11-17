using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Remind.DNForecast.Studio
{
    internal sealed partial class Server
    {
        private volatile string cudaDevice="",cudaProblem="Ověřuji zařízení…";
        private void ProbeCuda() {
            try {
                var psi=new ProcessStartInfo(trainer,"--cuda-device") {UseShellExecute=false,CreateNoWindow=true,
                    RedirectStandardOutput=true,RedirectStandardError=true};
                using(var p=Process.Start(psi)) {
                    var output=p.StandardOutput.ReadToEndAsync();var error=p.StandardError.ReadToEndAsync();
                    if(!p.WaitForExit(30000)){p.Kill();throw new IOException("Vypršel čas kontroly CUDA.");}
                    Task.WaitAll(output,error);
                    if(p.ExitCode!=0)throw new IOException(error.Result.Trim());
                    cudaDevice=output.Result.Trim();cudaProblem="";
                }
            } catch(Exception e){cudaProblem=e.Message;}
        }
        private static string ReadGraph(Stream input,int length) {
            if(length>256*1024)throw new InvalidDataException("Popis sítě překročil 256 KiB.");
            byte[] data=new byte[length];
            for(int i=0;i<length;) {int n=input.Read(data,i,length-i);if(n<=0)throw new EndOfStreamException();i+=n;}
            return Encoding.UTF8.GetString(data).Trim();
        }
    }

    internal static class Architecture
    {
        public static void Validate(string graph,int markets,int horizons)
        {
            var widths=new List<int>();var parents=new List<int[]>();
            var invariant=CultureInfo.InvariantCulture;
            long parameters=0;
            foreach(string line in graph.Split(new[]{'\n'},StringSplitOptions.RemoveEmptyEntries)) {
                var v=line.Split((char[])null,StringSplitOptions.RemoveEmptyEntries);
                int w;double drop;
                if(widths.Count>=1024 || v.Length<6 || v.Length>69 || !new[]{"LSTM","GRU","RNN","Dense","Identity"}.Contains(v[0]) ||
                    !int.TryParse(v[1],NumberStyles.None,invariant,out w) || !new[]{"linear","tanh","relu","sigmoid"}.Contains(v[2]) ||
                    (v[0]!="Dense" && v[2]!="linear") || !double.TryParse(v[3],NumberStyles.Float,invariant,out drop) ||
                    !(drop>=0 && (float)drop<1) || (v[4]!="sum" && v[4]!="concat")) throw new InvalidDataException("Neplatná vrstva "+(widths.Count+1));
                var edges=new List<int>();long input=0;
                foreach(string text in v.Skip(5)) {
                    int p;if(!int.TryParse(text,NumberStyles.AllowLeadingSign,invariant,out p) || p< -1 || p>=widths.Count)
                        throw new InvalidDataException("Vstup vrstvy musí být vstupní sekvence nebo předchozí vrstva.");
                    int n=p==-1?3*markets:widths[p];
                    if(v[4]=="sum" && input>0 && n!=input)throw new InvalidDataException("Součet větví vyžaduje stejnou šířku.");
                    input=v[4]=="sum"?n:input+n;edges.Add(p);
                }
                if(input>Int32.MaxValue)throw new InvalidDataException("Příliš široký vstup.");
                if(v[0]=="Identity" && w==0)w=(int)input;
                if(w<1 || (v[0]=="Identity" && w!=input))throw new InvalidDataException("Nesprávná šířka vrstvy.");
                int gates=v[0]=="LSTM"?4:v[0]=="GRU"?3:v[0]=="Identity"?0:1;
                bool recurrent=v[0]=="LSTM" || v[0]=="GRU" || v[0]=="RNN";
                if(w>16777216 || input>16777216)throw new InvalidDataException("Příliš široká vrstva.");
                parameters+=(input+(recurrent?w:0)+1)*gates*w+(v[0]=="GRU"?w:0);
                if(parameters>16777216)throw new InvalidDataException("Model překračuje limit 64 MiB pro MT5.");
                widths.Add(w);parents.Add(edges.ToArray());
            }
            if(widths.Count==0)throw new InvalidDataException("Přidejte alespoň jednu vrstvu.");
            if(parameters+((long)widths.Last()+1)*horizons>16777216)throw new InvalidDataException("Výstup překračuje limit modelu MT5.");
            var used=new bool[widths.Count];used[used.Length-1]=true;
            for(int i=used.Length-1;i>=0;i--)if(used[i])foreach(int p in parents[i])if(p>=0)used[p]=true;
            if(used.Contains(false))throw new InvalidDataException("Každá větev musí vést k poslední vrstvě.");
        }
    }
}
