using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: AssemblyTitle("DNForecast Setup for MetaTrader 5")]
[assembly: AssemblyProduct("DNForecast")]
[assembly: AssemblyCompany("Remind / Tomáš Bělák")]
[assembly: AssemblyCopyright("Tomáš Bělák / Remind")]
[assembly: AssemblyVersion("3.0.0.1")]
[assembly: AssemblyFileVersion("3.0.0.1")]

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

        private static int Install(Options args, Action<string> update, CancellationToken cancel)
        {
            string logfile = Path.GetFullPath(args.Log ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Remind", "DNForecast", "Logs",
                "Setup-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N") + ".log"));
            Directory.CreateDirectory(Path.GetDirectoryName(logfile));
            using (var writer = new StreamWriter(logfile, true))
            {
                writer.AutoFlush = true;
                Action<string> log = s => writer.WriteLine("{0:O}  {1}", DateTime.Now, s);
                List<string> pending;
                for (;;)
                {
                    pending = args.Target == null
                        ? Terminals.Discover(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                            AppDomain.CurrentDomain.BaseDirectory, Terminals.RunningDirectories(), log)
                        : new List<string> { Terminals.Explicit(args.Target) };
                    if (pending.Count != 0) break;
                    if (args.Quiet || args.NoWait) { log("No initialized MT5 data directory found."); return 2; }
                    update(T("Čekám na první spuštění MetaTraderu 5. Otevřete svůj terminál a instalace bude automaticky pokračovat.",
                             "Waiting for MetaTrader 5. Open your terminal once and installation will continue automatically."));
                    if (cancel.WaitHandle.WaitOne(2000)) cancel.ThrowIfCancellationRequested();
                }
                int total = pending.Count, errors = 0, finished = 0;
                var reported = new HashSet<string>();
                using (var zip = Installation.OpenPayload())
                while (pending.Count != 0)
                {
                    foreach (string target in pending.ToArray())
                    {
                        cancel.ThrowIfCancellationRequested();
                        if (!reported.Contains(target))
                            update(T("Instaluji do ", "Installing into ") + target);
                        try
                        {
                            bool changed = Installation.Apply(target, zip, log, Environment.SystemDirectory);
                            log((changed ? "Installed: " : "Already current: ") + target);
                            ++finished;
                        }
                        catch (IOException e)
                        {
                            int code = e.HResult & 0xffff;
                            if (code == 32 || code == 33 || code == 1224)
                            {
                                if (reported.Add(target)) log("Waiting for file: " + e.Message);
                                continue;
                            }
                            ++errors; log("Error: " + target + " | " + e.Message);
                        }
                        catch (UnauthorizedAccessException e) { ++errors; log("Access: " + target + " | " + e.Message); }
                        pending.Remove(target);
                    }
                    if (pending.Count == 0) break;
                    update(T("Soubor je právě používán. Instalace bude automaticky pokračovat po jeho uvolnění. Pokud jej používá MT5, zavřete terminál, až vám to bude vyhovovat.",
                             "A file is in use. Setup will continue automatically when it is released. If MT5 is using it, close the terminal when convenient.")
                           + Environment.NewLine + pending[0]);
                    if (args.NoWait) return 32;
                    if (cancel.WaitHandle.WaitOne(1500)) cancel.ThrowIfCancellationRequested();
                }
                if (errors != 0)
                {
                    update(T("Dokončeno pro ", "Completed for ") + finished + " / " + total +
                           T(" instalací. Podrobnosti o chybě: ", " installations. See the error log: ") + logfile);
                    return 1;
                }
                update(T("DNForecast je připraven v MT5", "DNForecast is installed in MT5") + " (" + finished + ")." +
                       Environment.NewLine + T("V Navigátoru obnovte seznam Indikátory a na graf přidejte DNForecast_Async.",
                                                "Refresh Indicators in Navigator and attach DNForecast_Async to a chart."));
                return 0;
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
                if (options.Quiet) return Install(options, s => { }, CancellationToken.None);
                Application.EnableVisualStyles();
                using (var form = new Form { Text = "DNForecast · MetaTrader 5", ClientSize = new Size(510, 162),
                        AutoScaleMode = AutoScaleMode.Dpi, FormBorderStyle = FormBorderStyle.FixedDialog,
                        MaximizeBox = false, StartPosition = FormStartPosition.CenterScreen })
                using (var token = new CancellationTokenSource())
                using (var timer = new System.Windows.Forms.Timer { Interval = 4500 })
                {
                    var title = new Label { Text = T("Instalace DNForecast", "Installing DNForecast"), Left = 20, Top = 15,
                        Width = 466, Height = 28, Font = new Font("Segoe UI", 14, FontStyle.Bold) };
                    var status = new Label { Left = 22, Top = 52, Width = 466, Height = 75, AutoEllipsis = true,
                        Text = T("Vyhledávám MetaTrader 5…", "Finding MetaTrader 5…") };
                    var progress = new ProgressBar { Left = 22, Top = 133, Width = 466, Height = 9, Style = ProgressBarStyle.Marquee };
                    form.Controls.AddRange(new Control[] { title, status, progress });
                    bool done = false;
                    int result = 1;
                    form.FormClosing += (sender, ev) => { if (!done) { token.Cancel(); ev.Cancel = true; } };
                    timer.Tick += (sender, ev) => { timer.Stop(); form.Close(); };
                    form.Shown += async (sender, ev) =>
                    {
                        IProgress<string> messages = new Progress<string>(s => status.Text = s);
                        try { result = await Task.Run(() => Install(options, messages.Report, token.Token)); }
                        catch (OperationCanceledException) { result = 1223; }
                        catch (Exception e) { status.Text = e.Message; result = 1; }
                        done = true;
                        progress.Visible = false;
                        title.Text = result == 0 ? T("Instalace dokončena", "Installation complete")
                                                : T("Stav instalace", "Installation status");
                        if (result == 1223) form.Close();
                        else if (result == 0) timer.Start();
                    };
                    Application.Run(form);
                    return result;
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
