using System;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("DNForecast Studio")]
[assembly: AssemblyCompany("Remind / Tomáš Bělák")]
[assembly: AssemblyVersion("3.3.0.0")]
namespace Remind.DNForecast.Studio
{
    internal static class Program
    {
        private static void Open(string url) { Process.Start(new ProcessStartInfo(url) { UseShellExecute=true }); }
        [STAThread]
        private static int Main(string[] args)
        {
            bool headless=Array.IndexOf(args,"--headless")>=0;
            try {
                string root=Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,@"..\Files\DNForecast"));
                string mt5Folder=null;
                int port=18765;
                for(int i=0;i<args.Length;i++) {
                    if(args[i]=="--headless") continue;
                    if(args[i]=="--data" && i+1<args.Length) root=Path.GetFullPath(args[++i]);
                    else if(args[i]=="--mt5-data" && i+1<args.Length) {
                        mt5Folder=Path.GetFullPath(args[++i]);
                        if(!Directory.Exists(Path.Combine(mt5Folder,"MQL5"))) throw new ArgumentException("Missing MQL5 directory in --mt5-data");
                    }
                    else if(args[i]=="--port" && i+1<args.Length) port=int.Parse(args[++i],CultureInfo.InvariantCulture);
                    else throw new ArgumentException("Options: --data PATH --mt5-data TERMINAL_DATA --port NUMBER --headless");
                }
                if(port<1 || port>65535) throw new ArgumentException("Port must be between 1 and 65535");
                string id;
                using(var sha=SHA256.Create()) id=BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(root.ToUpperInvariant()))).Replace("-","");
                bool created;
                using(var instance=new Mutex(false,@"Local\DNForecast-Studio-"+id,out created)) {
                    if(!created) {
                        if(headless) throw new IOException("Studio already owns this data directory.");
                        string location=File.ReadAllText(Path.Combine(root,"studio-port.txt"));
                        Open("http://127.0.0.1:"+int.Parse(location,CultureInfo.InvariantCulture)+"/");return 0;
                    }
                    using(var server=new Server(root,port,mt5Folder)) {
                        File.WriteAllText(Path.Combine(root,"studio-port.txt"),port.ToString(CultureInfo.InvariantCulture));
                        if(headless) { Thread.Sleep(Timeout.Infinite);return 0; }
                        Application.EnableVisualStyles();
                        using(var menu=new ContextMenuStrip())
                        using(var tray=new NotifyIcon { Icon=Icon.ExtractAssociatedIcon(Application.ExecutablePath),Text="DNForecast Studio",ContextMenuStrip=menu }) {
                            menu.Items.Add("Otevřít studio",null,(s,e)=>Open(server.Url));
                            menu.Items.Add("Ukončit server",null,(s,e)=>{
                                if(server.Busy) MessageBox.Show("Probíhá trénink. Server můžete ukončit po dokončení úlohy.","DNForecast Studio");
                                else Application.ExitThread();
                            });
                            tray.DoubleClick+=(s,e)=>Open(server.Url);tray.Visible=true;
                            Open(server.Url);Application.Run();tray.Visible=false;
                        }
                    }
                }
                return 0;
            } catch(Exception e) {
                if(!headless) MessageBox.Show(e.Message,"DNForecast Studio",MessageBoxButtons.OK,MessageBoxIcon.Error);
                else Console.Error.WriteLine(e.Message);
                return 1;
            }
        }
    }
}
