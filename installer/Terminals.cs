using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace Remind.DNForecast.Setup
{
    internal static class Terminals
    {
        private static bool HasTerminal(string path)
        {
            return File.Exists(Path.Combine(path, "terminal64.exe"));
        }

        private static bool HasData(string path)
        {
            return Directory.Exists(Path.Combine(path, "MQL5"));
        }

        internal static string Explicit(string path)
        {
            string root = Path.GetFullPath(path);
            if (!HasData(root))
                throw new DirectoryNotFoundException("Missing MQL5 data directory: " + root);
            return root;
        }

        internal static IEnumerable<string> RunningDirectories()
        {
            var paths = new List<string>();
            using (var current = Process.GetCurrentProcess())
            foreach (var process in Process.GetProcessesByName("terminal64"))
            using (process)
            {
                try
                {
                    if (process.SessionId == current.SessionId)
                        paths.Add(Path.GetDirectoryName(process.MainModule.FileName));
                }
                catch (System.ComponentModel.Win32Exception) { } // Another user or elevated terminal.
                catch (InvalidOperationException) { }             // Terminal just exited.
            }
            return paths;
        }

        internal static List<string> Discover(string roaming, string beside, IEnumerable<string> running,
                                               Action<string> log)
        {
            // Placing Setup beside a portable terminal explicitly selects only that copy.
            if (HasTerminal(beside) && HasData(beside)) return new List<string> { beside };
            var roots = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
            var standardPrograms = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string profiles = Path.Combine(roaming, "MetaQuotes", "Terminal");
            if (Directory.Exists(profiles))
            foreach (string folder in Directory.GetDirectories(profiles))
            {
                string name = Path.GetFileName(folder);
                if (name.Length != 32 || !name.All(Uri.IsHexDigit)) continue;
                try
                {
                    string origin = Path.Combine(folder, "origin.txt");
                    if (!File.Exists(origin)) continue;
                    // MetaTrader writes UTF-16 LE, sometimes without a BOM.
                    byte[] bytes = File.ReadAllBytes(origin);
                    string path = (bytes.Length > 1 && (bytes[0] == 0xff || bytes[1] == 0))
                        ? Encoding.Unicode.GetString(bytes) : Encoding.UTF8.GetString(bytes);
                    path = path.Trim('\ufeff', '\0', ' ', '\r', '\n', '\t');
                    if (path.Length == 0 || !Path.IsPathRooted(path) || !HasTerminal(path)) continue;
                    if (HasData(folder))
                    {
                        roots.Add(Path.GetFullPath(folder));
                        standardPrograms.Add(Path.GetFullPath(path));
                    }
                    // Some brokers ship seed MQL5 files under Program Files even in main mode.
                    if (HasData(path) && (!HasData(folder) || Directory.Exists(Path.Combine(path, "MQL5", "Logs"))))
                        roots.Add(Path.GetFullPath(path));
                }
                catch (IOException error) { log(folder + ": " + error.Message); }
                catch (UnauthorizedAccessException error) { log(folder + ": " + error.Message); }
                catch (ArgumentException error) { log(folder + ": " + error.Message); }
            }
            foreach (string path in running)
                if (!standardPrograms.Contains(Path.GetFullPath(path)) && HasTerminal(path) && HasData(path))
                    roots.Add(Path.GetFullPath(path));
            return roots.ToList();
        }
    }
}
