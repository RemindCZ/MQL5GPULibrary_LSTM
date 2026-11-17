using System;
using System.Collections.Generic;
using System.Globalization;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Remind.DNForecast.Setup
{
    // The build derives the embedded payload and its allowlist from the verified release.
    internal sealed class Item
    {
        internal readonly string Source, Target, Digest;
        internal readonly Version RuntimeVersion;
        internal Item(string source, string target, string digest, string runtimeVersion = null)
        {
            Source = source;
            Target = target;
            Digest = digest;
            RuntimeVersion = runtimeVersion == null ? null : Version.Parse(runtimeVersion);
        }

        internal bool RuntimeAvailable(string destination, string systemDirectory)
        {
            if (RuntimeVersion == null) return false;
            string candidate = File.Exists(destination) ? destination : Path.Combine(systemDirectory, Path.GetFileName(destination));
            if (!File.Exists(candidate)) return false;
            var v = FileVersionInfo.GetVersionInfo(candidate);
            return new Version(v.FileMajorPart, v.FileMinorPart, v.FileBuildPart, v.FilePrivatePart) >= RuntimeVersion;
        }
    }

    internal static class Installation
    {
        private const string Workspace = ".DNForecast-Setup-v1";
        private static string Hash(Stream stream)
        {
            using (var sha = SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        private static string HashFile(string path)
        {
            using (var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete))
                return Hash(input);
        }

        internal static ZipArchive OpenPayload()
        {
            Stream zip = Assembly.GetExecutingAssembly().GetManifestResourceStream("release.zip");
            if (zip == null) throw new InvalidDataException("Missing installer payload.");
            try
            {
                if (Hash(zip) != Payload.ArchiveHash) throw new InvalidDataException("Invalid installer payload digest.");
                zip.Position = 0;
                return new ZipArchive(zip, ZipArchiveMode.Read);
            }
            catch { zip.Dispose(); throw; }
        }

        private static void RejectLink(string path)
        {
            try
            {
                if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidDataException("A package destination is a link: " + path);
            }
            catch (FileNotFoundException) { }
            catch (DirectoryNotFoundException) { }
        }

        private static string Inside(string root, string relative)
        {
            string path = root;
            foreach (string part in relative.Split('/'))
            {
                if (part == "" || part == "." || part == ".." || part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                    throw new InvalidDataException("Invalid destination " + relative);
                path = Path.Combine(path, part);
                RejectLink(path);
            }
            return path;
        }

        private static void WriteNew(string path, Stream input)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            using (var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                input.CopyTo(file);
                file.Flush(true);
            }
        }

        private static void Mark(string path, IEnumerable<string> rows)
        {
            string temp = path + ".tmp";
            using (var data = new MemoryStream(Encoding.UTF8.GetBytes(string.Join("\n", rows))))
                WriteNew(temp, data);
            File.Move(temp, path);
        }

        private static void Discard(string work)
        {
            // Delete only known staging files. Never recursively remove an existing user's directory.
            foreach (var item in Payload.Items)
            foreach (string prefix in new[] { "new/", "old/" })
            {
                string path = Inside(work, prefix + item.Target);
                if (File.Exists(path)) File.Delete(path);
                string folder = Path.GetDirectoryName(path);
                while (folder != work && Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any())
                {
                    Directory.Delete(folder);
                    folder = Path.GetDirectoryName(folder);
                }
            }
            foreach (string name in new[] { "restore", "prepared.tmp", "prepared", "complete.tmp", "complete" })
                File.Delete(Inside(work, name));
            Directory.Delete(work); // A foreign file keeps the directory intact.
        }

        private static void Recover(string root, string work, Action<string> log)
        {
            if (!Directory.Exists(work)) return;
            string complete = Inside(work, "complete");
            string plan = Inside(work, "prepared");
            if (File.Exists(complete))
            {
                string old = Inside(work, "old");
                if (Directory.Exists(old))
                {
                    string backup = Inside(root, "DNForecast.Backups/" +
                        DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture) + "-" +
                        Guid.NewGuid().ToString("N"));
                    Directory.CreateDirectory(Path.GetDirectoryName(backup));
                    Directory.Move(old, backup);
                    log("Backup: " + backup);
                }
            }
            else if (File.Exists(plan))
            {
                log("Restoring previous files in " + root);
                foreach (string line in File.ReadAllLines(plan).Reverse())
                {
                    string[] parts = line.Split('\t');
                    if (parts.Length != 3) throw new InvalidDataException("Invalid recovery record in " + work);
                    Item item = Payload.Items.Single(x => x.Target == parts[0]);
                    string staged = Inside(work, "new/" + item.Target);
                    if (File.Exists(staged)) continue; // Atomic replacement did not take place.
                    string dest = Inside(root, item.Target);
                    string actual = File.Exists(dest) ? HashFile(dest) : "-";
                    if (actual == parts[1]) continue;
                    if (actual != parts[2]) throw new IOException("File changed during recovery: " + dest);
                    if (parts[1] == "-") File.Delete(dest);
                    else
                    {
                        string old = Inside(work, "old/" + item.Target);
                        if (HashFile(old) != parts[1]) throw new InvalidDataException("Invalid backup: " + old);
                        string restore = Inside(work, "restore");
                        File.Copy(old, restore, true);
                        File.Replace(restore, dest, null);
                    }
                }
            }
            Discard(work);
        }

        internal static bool Apply(string root, ZipArchive zip, Action<string> log, string systemDirectory)
        {
            string gate = Inside(root, ".DNForecast-Setup.lock");
            using (new FileStream(gate, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None,
                                  4096, FileOptions.DeleteOnClose))
            {
                string work = Inside(root, Workspace);
                Recover(root, work, log);
                var changed = new List<Item>();
                var plan = new List<string>();
                try
                {
                    foreach (Item item in Payload.Items)
                    {
                        string dest = Inside(root, item.Target);
                        if (item.RuntimeAvailable(dest, systemDirectory)) continue;
                        string previous = File.Exists(dest) ? HashFile(dest) : "-";
                        if (previous == item.Digest) continue;
                        var entry = zip.GetEntry(item.Source);
                        if (entry == null) throw new InvalidDataException("Missing " + item.Source);
                        string stage = Inside(work, "new/" + item.Target);
                        using (var input = entry.Open()) WriteNew(stage, input);
                        if (HashFile(stage) != item.Digest) throw new InvalidDataException("Invalid file: " + item.Source);
                        if (previous != "-")
                        {
                            string old = Inside(work, "old/" + item.Target);
                            // Detect files in use before committing any change to this terminal.
                            using (var existing = new FileStream(dest, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                                WriteNew(old, existing);
                            if (HashFile(old) != previous) throw new IOException("File changed during backup: " + dest);
                        }
                        plan.Add(item.Target + "\t" + previous + "\t" + item.Digest);
                        changed.Add(item);
                    }
                    if (changed.Count == 0) return false;
                    Mark(Inside(work, "prepared"), plan);
                    for (int i = 0; i < changed.Count; ++i)
                    {
                        Item item = changed[i];
                        string dest = Inside(root, item.Target);
                        string expected = plan[i].Split('\t')[1];
                        string stage = Inside(work, "new/" + item.Target);
                        Directory.CreateDirectory(Path.GetDirectoryName(dest));
                        if (expected == "-") File.Move(stage, dest);
                        else
                        {
                            if (HashFile(dest) != expected) throw new IOException("File changed during setup: " + dest);
                            File.Replace(stage, dest, null);
                        }
                        if (HashFile(dest) != item.Digest) throw new InvalidDataException("Verification failed: " + dest);
                    }
                    Mark(Inside(work, "complete"), new[] { "OK" });
                }
                catch
                {
                    Recover(root, work, log);
                    throw;
                }
                Recover(root, work, log);
                return true;
            }
        }
    }
}
