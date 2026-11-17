using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security;
using System.Text;
using Microsoft.Win32;

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
            var directory = new DirectoryInfo(root);
            if (string.Equals(directory.Name, "MQL5", StringComparison.OrdinalIgnoreCase))
                root = directory.Parent.FullName;
            if (!HasData(root))
                throw new DirectoryNotFoundException("Missing MQL5 data directory: " + root);
            return root;
        }

        internal static string Name(string folder)
        {
            string origin = Path.Combine(folder, "origin.txt");
            if (File.Exists(origin))
            {
                try
                {
                    string path = ReadOrigin(origin);
                    if (Path.IsPathRooted(path)) return new DirectoryInfo(path).Name;
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
                catch (ArgumentException) { }
            }
            return new DirectoryInfo(folder).Name;
        }

        private static string ReadOrigin(string origin)
        {
            byte[] bytes = File.ReadAllBytes(origin);
            Encoding encoding = bytes.Length > 1 && (bytes[0] == 0xff || bytes[1] == 0) ? Encoding.Unicode : Encoding.UTF8;
            return encoding.GetString(bytes).Trim('\ufeff', '\0', ' ', '\r', '\n', '\t');
        }

        internal static string[] ProgramRoots()
        {
            return new[] { Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs") };
        }

        internal static HashSet<string> RegisteredPrograms()
        {
            var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (RegistryHive hive in new[] { RegistryHive.CurrentUser, RegistryHive.LocalMachine })
            foreach (RegistryView view in new[] { RegistryView.Registry64, RegistryView.Registry32 })
            {
                try
                {
                    using (var registry = RegistryKey.OpenBaseKey(hive, view))
                    using (var programs = registry.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall"))
                    {
                        if (programs == null) continue;
                        foreach (string name in programs.GetSubKeyNames())
                        {
                            try
                            {
                                using (var entry = programs.OpenSubKey(name))
                                {
                                    if (entry == null) continue;
                                    string location = entry.GetValue("InstallLocation") as string;
                                    if (string.IsNullOrWhiteSpace(location)) continue;
                                    location = location.Trim().Trim('"');
                                    if (Path.IsPathRooted(location))
                                        paths.Add(Path.GetFullPath(location).TrimEnd(Path.DirectorySeparatorChar));
                                }
                            }
                            catch (IOException) { }
                            catch (UnauthorizedAccessException) { }
                            catch (SecurityException) { }
                            catch (ArgumentException) { }
                            catch (NotSupportedException) { }
                        }
                    }
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
                catch (SecurityException) { }
            }
            return paths;
        }

        internal static List<string> Discover(string roaming, string[] programRoots, HashSet<string> registered,
                                               Action<string> log)
        {
            var roots = new List<string>();
            string[] prefixes = programRoots.Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(p => Path.GetFullPath(p).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar).ToArray();
            string profiles = Path.Combine(roaming, "MetaQuotes", "Terminal");
            if (Directory.Exists(profiles))
            foreach (string folder in Directory.GetDirectories(profiles))
            {
                string name = Path.GetFileName(folder);
                if (name.Length != 32 || !name.All(Uri.IsHexDigit)) continue;
                try
                {
                    if (!HasData(folder) || (File.GetAttributes(folder) & FileAttributes.ReparsePoint) != 0 ||
                        (File.GetAttributes(Path.Combine(folder, "MQL5")) & FileAttributes.ReparsePoint) != 0) continue;
                    string origin = Path.Combine(folder, "origin.txt");
                    if (!File.Exists(origin)) continue;
                    // MetaTrader writes UTF-16 LE, sometimes without a BOM.
                    string path = ReadOrigin(origin);
                    if (path.Length == 0 || !Path.IsPathRooted(path) || !HasTerminal(path)) continue;
                    path = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar);
                    if (registered.Contains(path) || prefixes.Any(p => path.StartsWith(p, StringComparison.OrdinalIgnoreCase)))
                        roots.Add(Path.GetFullPath(folder));
                }
                catch (IOException error) { log(folder + ": " + error.Message); }
                catch (UnauthorizedAccessException error) { log(folder + ": " + error.Message); }
                catch (ArgumentException error) { log(folder + ": " + error.Message); }
            }
            roots.Sort(StringComparer.OrdinalIgnoreCase);
            return roots;
        }
    }
}
