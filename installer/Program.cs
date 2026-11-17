using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("DNForecast Setup for MetaTrader 5")]
[assembly: AssemblyProduct("DNForecast")]
[assembly: AssemblyCompany("Remind / Tomáš Bělák")]
[assembly: AssemblyCopyright("Tomáš Bělák / Remind")]
[assembly: AssemblyVersion("3.0.0.2")]
[assembly: AssemblyFileVersion("3.0.0.2")]

namespace Remind.DNForecast.Setup
{
    internal static class Program
    {
        internal static string T(string cs, string en)
        {
            return CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "cs" ? cs : en;
        }

        private sealed class Options
        {
            internal string Target, Log;
            internal bool Quiet, NoWait;
            internal Options(string[] args)
            {
                for (int i = 0; i < args.Length; ++i)
                {
                    switch (args[i].TrimStart('-', '/').ToLowerInvariant())
                    {
                        case "quiet": case "silent": Quiet = true; break;
                        case "no-wait": NoWait = true; break;
                        case "target": if (++i == args.Length) throw new ArgumentException("/target DATA_FOLDER"); Target = args[i]; break;
                        case "log": if (++i == args.Length) throw new ArgumentException("/log FILE"); Log = args[i]; break;
                        default: throw new ArgumentException("Unknown argument: " + args[i]);
                    }
                }
            }
        }

        internal static List<string> Candidates()
        {
            return Terminals.Discover(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                Terminals.ProgramRoots(), Terminals.RegisteredPrograms(), s => Trace.WriteLine(s));
        }

        internal static int Install(string target, string logPath, bool noWait, Action<string> update, CancellationToken cancel)
        {
            string root = Terminals.Explicit(target);
            string logfile = Path.GetFullPath(logPath ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Remind", "DNForecast", "Logs",
                "Setup-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N") + ".log"));
            Directory.CreateDirectory(Path.GetDirectoryName(logfile));
            using (var writer = new StreamWriter(logfile, true))
            {
                writer.AutoFlush = true;
                Action<string> log = s => writer.WriteLine("{0:O}  {1}", DateTime.Now, s);
                log("Target: " + root);
                try
                {
                    using (var zip = Installation.OpenPayload())
                    {
                        bool waiting = false;
                        for (;;)
                        {
                            cancel.ThrowIfCancellationRequested();
                            try
                            {
                                bool changed = Installation.Apply(root, zip, log, Environment.SystemDirectory);
                                log((changed ? "Installed: " : "Already current: ") + root);
                                return 0;
                            }
                            catch (IOException e)
                            {
                                int code = e.HResult & 0xffff;
                                if (code != 32 && code != 33 && code != 1224) throw;
                                if (!waiting) log("Waiting for file: " + e.Message);
                                waiting = true;
                                update(T("Soubor používá jiný proces. Po jeho uvolnění bude instalace pokračovat.",
                                         "A file is in use. Setup will continue when it is released."));
                                if (noWait) return 32;
                                if (cancel.WaitHandle.WaitOne(1500)) cancel.ThrowIfCancellationRequested();
                            }
                        }
                    }
                }
                catch (OperationCanceledException) { throw; }
                catch (Exception e)
                {
                    log("Error: " + e.Message);
                    update(e.Message + Environment.NewLine + logfile);
                    return 1;
                }
            }
        }

        [STAThread]
        private static int Main(string[] args)
        {
            bool quiet = args.Any(a => string.Equals(a.TrimStart('-', '/'), "quiet", StringComparison.OrdinalIgnoreCase)
                                    || string.Equals(a.TrimStart('-', '/'), "silent", StringComparison.OrdinalIgnoreCase));
            try
            {
                var options = new Options(args);
                if (options.Quiet)
                {
                    string target = options.Target;
                    if (target == null)
                    {
                        var candidates = Candidates();
                        if (candidates.Count == 0) return 2;
                        if (candidates.Count > 1) return 3;
                        target = candidates[0];
                    }
                    return Install(target, options.Log, options.NoWait, s => { }, CancellationToken.None);
                }
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                using (var form = new InstallerWindow(Candidates(), options.Target, options.Log, options.NoWait))
                {
                    Application.Run(form);
                    return form.Result;
                }
            }
            catch (Exception error)
            {
                if (!quiet) MessageBox.Show(error.Message, "DNForecast", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
    }
}
