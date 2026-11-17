using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web;
using System.Web.Script.Serialization;

namespace Remind.DNForecast.Studio
{
    internal sealed class Job
    {
        public string id, dataset, status, model, log, contract;
    }
    internal sealed class Server : IDisposable
    {
        private const int MaxUpload = 24 * 1024 * 1024;
        private readonly string root, history, trainer, token;
        private readonly TcpListener listener;
        private readonly object gate = new object();
        private readonly List<Job> jobs = new List<Job>();
        private readonly SemaphoreSlim clients = new SemaphoreSlim(8);
        private readonly byte[] page, picture;
        private bool busy;
        private volatile bool stopping;
        internal string Url { get; private set; }
        internal bool Busy { get { lock (gate) return busy; } }
        internal Server(string folder, int port)
        {
            root = Path.GetFullPath(folder);
            history = Path.Combine(root, "History");
            Directory.CreateDirectory(history);
            trainer = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"DNForecast-Train.exe");
            if (!File.Exists(trainer)) throw new FileNotFoundException("Missing native trainer",trainer);
            foreach (var f in new DirectoryInfo(root).GetFiles("Studio-*.job.json").OrderByDescending(v=>v.LastWriteTimeUtc).Take(30)) {
                try {
                    if (f.Length>65536 || !RegularFile(f.FullName)) continue;
                    var j=new JavaScriptSerializer().Deserialize<Job>(File.ReadAllText(f.FullName));
                    if(j!=null && j.id!=null && Regex.IsMatch(j.id,"^[0-9a-f]{32}$") &&
                       j.model=="Studio-"+j.id+".onnx" && RegularFile(Path.Combine(root,j.model))) jobs.Add(j);
                } catch (Exception) { }
            }
            byte[] random = new byte[32]; using (var rng = RandomNumberGenerator.Create()) rng.GetBytes(random);
            token = BitConverter.ToString(random).Replace("-","");
            using (var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("index.html"))
            using (var reader = new StreamReader(s)) page = Encoding.UTF8.GetBytes(reader.ReadToEnd().Replace("__NONCE__",token));
            using (var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("header.jpg"))
            using (var mem = new MemoryStream()) { s.CopyTo(mem); picture = mem.ToArray(); }
            listener = new TcpListener(IPAddress.Loopback,port); listener.Start(16);
            Url = "http://127.0.0.1:" + ((IPEndPoint)listener.LocalEndpoint).Port + "/";
            Task.Run((Action)Accept);
        }
        private void Accept()
        {
            while (!stopping)
            {
                try {
                    var connection = listener.AcceptTcpClient();
                    if (!clients.Wait(0)) { connection.Close(); continue; }
                    Task.Run(() => { try { using (connection) Serve(connection); } catch (Exception) { } finally { clients.Release(); } });
                } catch (SocketException) { if (!stopping) throw; }
                  catch (ObjectDisposedException) { return; }
            }
        }
        private static string Line(Stream stream, ref int remaining)
        {
            var b = new StringBuilder();
            while (--remaining > 0) {
                int ch=stream.ReadByte();
                if (ch==-1) throw new EndOfStreamException();
                if (ch=='\r') { if (stream.ReadByte()!='\n' || --remaining<0) throw new InvalidDataException("Malformed HTTP"); return b.ToString(); }
                if (ch<32 || ch>126) throw new InvalidDataException("Invalid HTTP header");
                b.Append((char)ch);
            }
            throw new InvalidDataException("Headers too large");
        }
        private static void Reply(Stream stream,int status,string type,byte[] data)
        {
            string headers="HTTP/1.1 "+status+" "+(status==200?"OK":"Error")+"\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'\r\nContent-Type: "+type+"\r\nContent-Length: "+data.Length+"\r\n\r\n";
            byte[] h=Encoding.ASCII.GetBytes(headers); stream.Write(h,0,h.Length); stream.Write(data,0,data.Length); stream.Flush();
        }
        private static void Text(Stream s,int status,string msg) { Reply(s,status,"text/plain; charset=utf-8",Encoding.UTF8.GetBytes(msg)); }
        private static bool IsDataName(string n) {
            return n!=null && n.Length<=180 && Path.GetFileName(n)==n && !n.Contains(":") && !n.StartsWith(".") &&
                n.IndexOfAny(Path.GetInvalidFileNameChars())<0 && n.EndsWith(".csv",StringComparison.OrdinalIgnoreCase);
        }
        private static bool RegularFile(string p) { return File.Exists(p) && (File.GetAttributes(p)&FileAttributes.ReparsePoint)==0; }
        private void Serve(TcpClient client)
        {
            client.ReceiveTimeout=10000; client.SendTimeout=10000;
            using (var stream=client.GetStream())
            using (var input=new BufferedStream(stream,8192)) {
                try {
                    int limit=8192;
                    var first=Line(input,ref limit).Split(' ');
                    if (first.Length!=3 || first[2]!="HTTP/1.1" || !first[1].StartsWith("/") || first[1].StartsWith("//")) throw new InvalidDataException("Invalid request");
                    var headers=new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
                    for (;;) {
                        string l=Line(input,ref limit); if (l.Length==0) break;
                        int i=l.IndexOf(':'); if(i<=0 || headers.ContainsKey(l.Substring(0,i))) throw new InvalidDataException("Invalid headers");
                        headers.Add(l.Substring(0,i),l.Substring(i+1).Trim());
                    }
                    int port=((IPEndPoint)listener.LocalEndpoint).Port;
                    string host;
                    if (!headers.TryGetValue("Host",out host) || !(host=="127.0.0.1:"+port || host=="localhost:"+port)) { Text(stream,403,"Invalid host");return; }
                    string raw=first[1]; int split=raw.IndexOf('?');
                    string path=split<0?raw:raw.Substring(0,split);
                    var query=HttpUtility.ParseQueryString(split<0?"":raw.Substring(split+1));
                    if (first[0]=="GET") {
                        if(path=="/") Reply(stream,200,"text/html; charset=utf-8",page);
                        else if(path=="/header.jpg") Reply(stream,200,"image/jpeg",picture);
                        else if(path=="/api/state") {
                            var datasets=Directory.EnumerateFiles(history,"*.csv").Where(RegularFile).Select(Path.GetFileName).Where(IsDataName).OrderByDescending(n=>n).ToArray();
                            string json;
                            lock(gate) json=new JavaScriptSerializer().Serialize(new { datasets=datasets, busy=busy, jobs=jobs.ToArray() });
                            Reply(stream,200,"application/json; charset=utf-8",Encoding.UTF8.GetBytes(json));
                        } else if(path.StartsWith("/model/")) {
                            Job j; lock(gate) j=jobs.Find(x=>x.id==path.Substring(7) && x.model!=null);
                            if(j==null) Text(stream,404,"Model unavailable");
                            else Reply(stream,200,"application/octet-stream",File.ReadAllBytes(Path.Combine(root,j.model)));
                        } else Text(stream,404,"Not found");
                        return;
                    }
                    string key,origin;
                    if(first[0]!="POST" || !headers.TryGetValue("X-DNForecast-Key",out key) || key!=token ||
                       (headers.TryGetValue("Origin",out origin) && origin!="http://"+host)) { Text(stream,403,"Invalid request origin");return; }
                    string contentLength; int length;
                    if(headers.ContainsKey("Transfer-Encoding") || !headers.TryGetValue("Content-Length",out contentLength) ||
                       !int.TryParse(contentLength,NumberStyles.None,CultureInfo.InvariantCulture,out length) || length>MaxUpload)
                    { Text(stream,413,"Maximum CSV size is 24 MiB");return; }
                    if(path=="/api/import" && length>0 && IsDataName(query["name"])) {
                        string safe=Regex.Replace(Path.GetFileNameWithoutExtension(query["name"]),@"[^a-zA-Z0-9_-]","_");
                        string name=safe.Substring(0,Math.Min(safe.Length,96))+"-"+Guid.NewGuid().ToString("N")+".csv";
                        string output=Path.Combine(history,name),temp=output+".tmp";
                        try {
                            using (var file=new FileStream(temp,FileMode.CreateNew,FileAccess.Write,FileShare.None)) {
                                byte[] buffer=new byte[65536]; int left=length;
                                while(left>0) { int n=input.Read(buffer,0,Math.Min(left,buffer.Length)); if(n==0) throw new EndOfStreamException(); file.Write(buffer,0,n);left-=n; }
                            }
                            using(var r=File.OpenText(temp)) {
                                var header=r.ReadLine();
                                if(header==null || !header.StartsWith("close_time,available_at,",StringComparison.Ordinal)) throw new InvalidDataException("CSV must begin with close_time,available_at,TARGET");
                            }
                            File.Move(temp,output);
                        } finally { if(File.Exists(temp)) File.Delete(temp); }
                        Text(stream,200,name);return;
                    }
                    if(path!="/api/train" || length!=0 || !IsDataName(query["file"])) throw new InvalidDataException("Invalid dataset or endpoint");
                    string dataset=query["file"], filePath=Path.Combine(history,dataset);
                    if (!RegularFile(filePath)) { Text(stream,404,"Dataset unavailable");return; }
                    string parameters="";
                    foreach (var item in new[]{Tuple.Create("seconds",1,31536000),Tuple.Create("steps",16,1024),Tuple.Create("horizons",1,64),Tuple.Create("epochs",1,64)}) {
                        int n;
                        if(!int.TryParse(query[item.Item1],out n) || n<item.Item2 || n>item.Item3) throw new InvalidDataException("Invalid "+item.Item1);
                        parameters+=" "+n.ToString(CultureInfo.InvariantCulture);
                    }
                    Job job;
                    lock(gate) {
                        if(busy) { Text(stream,409,"Another training job is running");return; }
                        string symbols;
                        using (var r=File.OpenText(filePath)) {
                            var h=r.ReadLine();
                            if(h==null || !h.StartsWith("close_time,available_at,") || h.Length>2048) throw new InvalidDataException("Invalid CSV header");
                            symbols=h.Substring("close_time,available_at,".Length);
                        }
                        job=new Job {id=Guid.NewGuid().ToString("N"),dataset=dataset,status="Probíhá",
                            contract=symbols+" | "+query["seconds"]+" s | "+query["steps"]+" kroků | "+query["horizons"]+" horizontů"};
                        busy=true; jobs.Insert(0,job); if(jobs.Count>30)jobs.RemoveAt(jobs.Count-1);
                    }
                    Task.Run(()=>Train(job,filePath,parameters)); Text(stream,200,job.id);
                } catch (InvalidDataException e) {Text(stream,400,e.Message);}
                  catch (IOException) {Text(stream,400,"Read or write failed");}
                  catch (Exception) {Text(stream,500,"Studio could not complete the request");}
            }
        }
        private void Train(Job job,string dataset,string parameters)
        {
            string name="Studio-"+job.id+".onnx";
            try {
                var psi=new ProcessStartInfo(trainer,"\""+dataset+"\" \""+Path.Combine(root,name)+"\""+parameters) {
                    UseShellExecute=false, CreateNoWindow=true, RedirectStandardOutput=true, RedirectStandardError=true };
                using(var p=Process.Start(psi)) {
                    var stdout=p.StandardOutput.ReadToEndAsync(); var stderr=p.StandardError.ReadToEndAsync();
                    p.WaitForExit(); Task.WaitAll(stdout,stderr);
                    if(p.ExitCode!=0) throw new InvalidDataException(stderr.Result.Trim());
                    File.WriteAllText(Path.Combine(root,"Studio-"+job.id+".csv"),stdout.Result,Encoding.UTF8);
                    lock(gate) {
                        job.log=stdout.Result.Trim();job.model=name;job.status="Dokončeno";
                        string meta=Path.Combine(root,"Studio-"+job.id+".job.json");
                        File.WriteAllText(meta+".tmp",new JavaScriptSerializer().Serialize(job),Encoding.UTF8); File.Move(meta+".tmp",meta);
                    }
                }
            } catch(Exception e) { lock(gate) {job.status="Chyba";job.log=e.Message;} }
            finally {lock(gate) busy=false;}
        }
        public void Dispose() { stopping=true; listener.Stop(); }
    }
}
