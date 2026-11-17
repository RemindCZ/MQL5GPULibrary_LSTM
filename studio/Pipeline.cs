using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace Remind.DNForecast.Studio
{
    internal sealed class Recipe
    {
        public string target,drivers,side;
        public int seconds,steps,horizons,epochs,bars,minutes;
        public double cost,edge,allocation;

        internal static Recipe Parse(NameValueCollection q,bool broker)
        {
            var c=new Recipe {
                seconds=Integer(q,"seconds",1,31536000),steps=Integer(q,"steps",16,1024),
                horizons=Integer(q,"horizons",1,64),epochs=Integer(q,"epochs",1,64),
                side=q["side"]??"both",cost=Decimal(q,"cost",1,0,100),edge=Decimal(q,"edge",2,0,1000),
                allocation=Decimal(q,"allocation",100,1,100)
            };
            if(c.horizons>c.steps/2 || !(new[]{"long","short","both"}).Contains(c.side))
                throw new InvalidDataException("Neplatný horizont nebo směr pozice.");
            if(broker) {
                c.bars=Integer(q,"bars",512,20000);
                if(!(new[]{60,300,900,1800,3600,14400,86400}).Contains(c.seconds))
                    throw new InvalidDataException("Nepodporovaný časový rámec MT5.");
                var symbols=((q["target"]??"")+","+ (q["drivers"]??"")).TrimEnd(',').Split(',').Select(v=>v.Trim()).ToArray();
                if(symbols.Length>8 || symbols.Any(v=>!Regex.IsMatch(v,@"\A[A-Za-z0-9_+#.:-]{1,64}\z")) ||
                    symbols.Distinct(StringComparer.Ordinal).Count()!=symbols.Length)
                    throw new InvalidDataException("Zadejte různé platné symboly brokera.");
                if(c.bars*55/100-c.horizons-c.steps<64 || c.bars-c.bars*85/100-c.horizons<32 ||
                   (long)c.bars*(24*symbols.Length+c.horizons)>16000000)
                    throw new InvalidDataException("Upravte počet svíček, symbolů nebo délku vstupu pro oddělený test.");
                c.target=symbols[0];c.drivers=String.Join(",",symbols.Skip(1));
            }
            if(q["minutes"]!=null) c.minutes=Integer(q,"minutes",1,10080);
            return c;
        }
        private static int Integer(NameValueCollection q,string key,int min,int max)
        {
            int n;if(!int.TryParse(q[key],NumberStyles.None,CultureInfo.InvariantCulture,out n) || n<min || n>max)
                throw new InvalidDataException("Neplatný parametr "+key);
            return n;
        }
        private static double Decimal(NameValueCollection q,string key,double fallback,double min,double max)
        {
            if(q[key]==null) return fallback;
            double n;if(!double.TryParse(q[key],NumberStyles.AllowDecimalPoint,CultureInfo.InvariantCulture,out n) || !(n>=min && n<=max))
                throw new InvalidDataException("Neplatný parametr "+key);
            return n;
        }
        internal string Symbols {get {return target+(String.IsNullOrEmpty(drivers)?"":","+drivers);}}
    }
    internal sealed class Schedule
    {
        public Recipe recipe;
        public string fingerprint, next, message, broker;
    }
    internal sealed partial class Server
    {
        private readonly string bridge;
        private Timer clock;
        private Schedule plan;
        private string PlanPath {get {return Path.Combine(root,"studio-plan.json");}}
        private static string At(DateTime time) {return time.ToString("o",CultureInfo.InvariantCulture);}
        private static void AtomicText(string path,string value)
        {
            string tmp=path+"."+Guid.NewGuid().ToString("N")+".tmp";
            try {
                File.WriteAllText(tmp,value,new UTF8Encoding(false));
                if(File.Exists(path)) File.Replace(tmp,path,null);else File.Move(tmp,path);
            } finally {if(File.Exists(tmp)) File.Delete(tmp);}
        }
        private sealed class FeedState
        {
            public bool online;
            public string broker,last;
            public string[] symbols;
        }
        private FeedState Feed(out bool online)
        {
            string path=Path.Combine(bridge,"status.txt");online=false;
            string broker="",last=null;string[] symbols=new string[0];
            try {
                if(RegularFile(path) && new FileInfo(path).Length<=65536) {
                    DateTime seen=File.GetLastWriteTimeUtc(path);
                    var lines=File.ReadAllLines(path);
                    if(lines.Length>=3 && lines[0]=="DNBRIDGE1") {
                        last=At(seen);broker=lines[2];symbols=lines.Skip(3).Take(512).ToArray();
                        online=lines[1]=="online" && (DateTime.UtcNow-seen).Duration()<TimeSpan.FromSeconds(20);
                    }
                }
            } catch(IOException) {}
            return new FeedState {online=online,broker=broker,last=last,symbols=symbols};
        }
        private void SavePlan() {AtomicText(PlanPath,new JavaScriptSerializer().Serialize(plan));}
        private void ResumePlan()
        {
            string marker=Path.Combine(root,"studio-pending.txt");
            if(RegularFile(marker)) {
                string id=File.ReadAllText(marker);
                if(Regex.IsMatch(id,@"\A[0-9a-f]{32}\z")) ClearRequest(Path.Combine(bridge,id));
                File.Delete(marker);
            }
            if(RegularFile(PlanPath)) plan=new JavaScriptSerializer().Deserialize<Schedule>(File.ReadAllText(PlanPath));
            clock=new Timer(_=>ScheduleTick(),null,1000,3000);
        }
        private void ScheduleTick()
        {
            try {
                lock(gate) {
                    if(stopping || busy || plan==null || DateTime.Parse(plan.next,null,DateTimeStyles.RoundtripKind)>DateTime.UtcNow) return;
                    bool online;Feed(out online);
                    if(!online) {plan.message="Čeká na připojení MT5";return;}
                    plan.next=At(DateTime.UtcNow.AddMinutes(plan.recipe.minutes));plan.message="Probíhá kontrola historie";SavePlan();
                    StartJob(plan.recipe,null,plan);
                }
            } catch(Exception e) { lock(gate) if(plan!=null) plan.message=e.Message; }
        }
        private Job StartJob(Recipe config,string dataset,Schedule scheduled)
        {
            var job=new Job {id=Guid.NewGuid().ToString("N"),dataset=dataset,status=dataset==null?"Načítání z MT5":"Trénink",
                config=config,created=At(DateTime.UtcNow),contract=String.Format("{0} | {1} s | {2} kroků | {3} horizontů",
                    config.target??dataset,config.seconds,config.steps,config.horizons)};
            jobs.Insert(0,job);busy=true;
            Task.Run(()=>Run(job,scheduled));return job;
        }
        private static void ClearRequest(string prefix)
        {
            foreach(string ext in new[]{".req",".csv",".err"}) File.Delete(prefix+ext);
        }
        private void Run(Job job,Schedule scheduled)
        {
            var c=job.config;string id=job.id, prefix=Path.Combine(bridge,id),hash=null;
            try {
                if(job.dataset==null) {
                    bool online;var feed=Feed(out online);
                    if(!online) throw new InvalidDataException("MT5 přerušil spojení.");
                    lock(gate) job.broker=feed.broker;
                    AtomicText(Path.Combine(root,"studio-pending.txt"),id);
                    AtomicText(prefix+".req",c.seconds+"|"+c.bars+"|"+c.Symbols);
                    var timeout=DateTime.UtcNow.AddSeconds(180);
                    while(!RegularFile(prefix+".csv")) {
                        if(RegularFile(prefix+".err")) throw new InvalidDataException(File.ReadAllText(prefix+".err"));
                        if(DateTime.UtcNow>=timeout || stopping) throw new InvalidDataException("MT5 nevrátil historii v časovém limitu.");
                        Thread.Sleep(250);
                    }
                    var received=Feed(out online);
                    if(received.broker!=job.broker) throw new InvalidDataException("Během přenosu se změnil server MT5.");
                    if(new FileInfo(prefix+".csv").Length>MaxUpload) throw new InvalidDataException("Historie je příliš velká.");
                    using(var r=File.OpenText(prefix+".csv")) {
                        if(r.ReadLine()!="close_time,available_at,"+c.Symbols) throw new InvalidDataException("Nesouhlasí symboly historie z MT5.");
                        int rows=0;while(r.ReadLine()!=null) rows++;
                        if(rows!=c.bars) throw new InvalidDataException("Neúplná historie MT5.");
                    }
                    using(var sha=SHA256.Create()) using(var stream=File.OpenRead(prefix+".csv"))
                        hash=BitConverter.ToString(sha.ComputeHash(stream)).Replace("-","");
                    if(scheduled!=null && hash==scheduled.fingerprint && scheduled.broker==job.broker) {
                        lock(gate) {job.status="Historie beze změny";if(plan==scheduled) plan.message=job.status;}
                        return;
                    }
                    string name=Regex.Replace(c.target,@"[^a-zA-Z0-9_-]","_")+"-"+c.seconds+"s-"+hash.Substring(0,20)+".csv";
                    string file=Path.Combine(history,name);
                    if(!RegularFile(file)) {
                        string staged=file+"."+id+".tmp";
                        try {File.Copy(prefix+".csv",staged);File.Move(staged,file);}
                        finally {if(File.Exists(staged)) File.Delete(staged);}
                    }
                    lock(gate) job.dataset=name;
                }
                string dataset=Path.Combine(history,job.dataset);
                string symbols;
                using(var r=File.OpenText(dataset)) {
                    string line=r.ReadLine();
                    if(line==null || line.Length>2048 || !line.StartsWith("close_time,available_at,",StringComparison.Ordinal)) throw new InvalidDataException("Neplatná hlavička CSV.");
                    symbols=line.Substring("close_time,available_at,".Length);
                }
                lock(gate) {
                    job.contract=symbols+" | "+c.seconds+" s | "+c.steps+" kroků | "+c.horizons+" horizontů";
                    job.status="Trénink";
                }
                Train(job,dataset);
                lock(gate) {
                    if(scheduled!=null && plan==scheduled) {plan.fingerprint=hash;plan.broker=job.broker;plan.message="Dokončeno";SavePlan();}
                }
            } catch(Exception e) {lock(gate) {job.status="Chyba";job.log=e.Message;if(scheduled!=null && plan==scheduled) plan.message=e.Message;}}
            finally {
                try {
                    if(c.target!=null) {ClearRequest(prefix);File.Delete(Path.Combine(root,"studio-pending.txt"));}
                } catch(IOException) {} catch(UnauthorizedAccessException) {}
                lock(gate) {
                    int completed=0,other=0;
                    jobs.RemoveAll(j=>j.model==null ? ++other>5 : ++completed>30);
                    busy=false;
                }
            }
        }
    }
}
