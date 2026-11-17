using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Remind.DNForecast.Setup
{
    internal sealed class InstallerWindow : Form
    {
        private readonly ComboBox terminals = new ComboBox();
        private readonly TextBox path = new TextBox();
        private readonly Label heading, message;
        private readonly Button browse, primary, cancel, studio;
        private readonly ProgressBar progress = new ProgressBar();
        private readonly CancellationTokenSource cancellation = new CancellationTokenSource();
        private readonly string logPath;
        private readonly bool noWait;
        private readonly List<string> folders = new List<string>();
        private readonly Image artwork;
        private bool installing, finished;
        internal int Result = 1223;

        private static string T(string cs, string en) { return Program.T(cs, en); }

        private static readonly Color Ink = Color.FromArgb(21, 33, 53);
        private static readonly Color Muted = Color.FromArgb(68, 83, 105);
        private static Label Caption(Control parent, string text, Rectangle bounds, Font font, Color color)
        {
            var label = new Label { Text = text, Bounds = bounds, Font = font, ForeColor = color,
                BackColor = Color.Transparent, AutoEllipsis = true, UseMnemonic = false };
            parent.Controls.Add(label);
            return label;
        }

        private static Button ActionButton(Control parent, string name, string label, Rectangle bounds, bool accent)
        {
            var button = new Button { Name = name, Text = label, Bounds = bounds, FlatStyle = FlatStyle.Flat,
                BackColor = accent ? Color.FromArgb(25, 78, 174) : Color.White,
                ForeColor = accent ? Color.White : Ink, Cursor = Cursors.Hand, UseVisualStyleBackColor = false };
            button.FlatAppearance.BorderColor = accent ? button.BackColor : Color.FromArgb(157, 170, 188);
            parent.Controls.Add(button);
            return button;
        }

        internal InstallerWindow(List<string> detected, string requested, string log, bool releaseFile)
        {
            logPath = log;
            noWait = releaseFile;
            Text = T("Instalace DNForecast", "DNForecast Setup");
            AutoScaleDimensions = new SizeF(96, 96);
            AutoScaleMode = AutoScaleMode.Dpi;
            Font = new Font("Segoe UI", 10);
            ClientSize = new Size(686, 496);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition = FormStartPosition.CenterScreen;
            MaximizeBox = false;
            BackColor = Color.White;
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);

            using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("setup-header.jpg"))
            using (var source = Image.FromStream(stream))
                artwork = new Bitmap(source);
            var header = new Panel { Bounds = new Rectangle(0, 0, 686, 148), BackColor = Color.FromArgb(11, 20, 43),
                BackgroundImage = artwork, BackgroundImageLayout = ImageLayout.Stretch };
            Controls.Add(header);
            Caption(header, "REMIND", new Rectangle(28, 17, 220, 20), new Font("Segoe UI", 9, FontStyle.Bold),
                Color.FromArgb(135, 222, 246));
            Caption(header, "DNForecast", new Rectangle(25, 38, 350, 63), new Font("Segoe UI", 33, FontStyle.Bold), Color.White);
            Caption(header, "MetaTrader 5   /   Windows x64   /   v3.1.0", new Rectangle(29, 106, 330, 23),
                Font, Color.FromArgb(213, 228, 247));

            heading = Caption(this, T("Kam chcete DNForecast nainstalovat?", "Where should DNForecast be installed?"),
                new Rectangle(27, 165, 633, 37), new Font("Segoe UI", 17, FontStyle.Bold), Ink);
            Caption(this, T("Terminál MetaTrader 5", "MetaTrader 5 terminal"), new Rectangle(29, 217, 620, 23), Font, Muted);
            terminals.Name = "Terminals";
            terminals.AccessibleName = T("Terminál MetaTrader 5", "MetaTrader 5 terminal");
            terminals.DropDownStyle = ComboBoxStyle.DropDownList;
            terminals.Bounds = new Rectangle(29, 242, 414, 30);
            terminals.SelectedIndexChanged += (sender, e) => ShowSelection();
            Controls.Add(terminals);
            browse = ActionButton(this, "Browse", T("Vybrat jinou složku…", "Choose another folder…"),
                new Rectangle(455, 239, 201, 34), false);
            browse.Click += (sender, e) => Browse();

            Caption(this, T("Cílová datová složka", "Destination data folder"),
                new Rectangle(29, 289, 620, 23), Font, Muted);
            path.Name = "TargetPath";
            path.AccessibleName = T("Cílová datová složka", "Destination data folder");
            path.Bounds = new Rectangle(30, 314, 626, 50);
            path.Font = new Font("Consolas", 10);
            path.ForeColor = Ink;
            path.ReadOnly = true;
            path.BackColor = Color.White;
            path.BorderStyle = BorderStyle.None;
            path.Multiline = true;
            path.ScrollBars = ScrollBars.Vertical;
            Controls.Add(path);
            message = Caption(this, "", new Rectangle(29, 367, 627, 47), Font, Muted);
            message.Name = "Status";

            var footer = new Panel { Bounds = new Rectangle(0, 418, 686, 78), BackColor = Color.FromArgb(243, 246, 250) };
            Controls.Add(footer);
            progress.Bounds = new Rectangle(29, 34, 287, 10);
            progress.Style = ProgressBarStyle.Marquee;
            progress.Visible = false;
            footer.Controls.Add(progress);
            cancel = ActionButton(footer, "Cancel", T("Zrušit", "Cancel"), new Rectangle(389, 20, 112, 39), false);
            studio = ActionButton(footer, "Studio", T("Otevřít studio", "Open studio"), new Rectangle(330, 20, 171, 39), false);
            studio.Visible = false;
            studio.Click += (sender, e) => {
                try { System.Diagnostics.Process.Start(Path.Combine(folders[terminals.SelectedIndex], @"MQL5\Libraries\DNForecast-Studio.exe")); }
                catch (Exception ex) { message.Text = ex.Message; }
            };
            primary = ActionButton(footer, "Install", T("Nainstalovat", "Install"), new Rectangle(513, 20, 143, 39), true);
            primary.Font = new Font("Segoe UI", 10, FontStyle.Bold);
            primary.Click += async (sender, e) => await StartInstallation();
            cancel.Click += (sender, e) => Close();
            AcceptButton = primary;
            CancelButton = cancel;
            FormClosing += (sender, e) =>
            {
                if (installing)
                {
                    cancellation.Cancel();
                    e.Cancel = true;
                }
            };

            foreach (string folder in detected) AddTarget(folder);
            if (requested != null)
            {
                try { terminals.SelectedIndex = AddTarget(Terminals.Explicit(requested)); }
                catch (Exception e) { ShowSelection(); message.Text = e.Message; }
            }
            else if (folders.Count == 1) terminals.SelectedIndex = 0;
            else ShowSelection();
        }

        private int AddTarget(string folder)
        {
            int index = folders.FindIndex(f => string.Equals(f, folder, StringComparison.OrdinalIgnoreCase));
            if (index >= 0) return index;
            string name = Terminals.Name(folder);
            bool duplicate = false;
            for (int i = 0; i < folders.Count; ++i)
            {
                if (!string.Equals(Terminals.Name(folders[i]), name, StringComparison.OrdinalIgnoreCase)) continue;
                terminals.Items[i] = name + "   ·   " + folders[i];
                duplicate = true;
            }
            folders.Add(folder);
            terminals.Items.Add(duplicate ? name + "   ·   " + folder : name);
            return folders.Count - 1;
        }

        private void ShowSelection()
        {
            int index = terminals.SelectedIndex;
            path.Text = index < 0 ? "" : folders[index];
            primary.Enabled = index >= 0;
            message.Text = index >= 0
                ? T("Nainstaluje se indikátor, model, nativní trenér a webové studio.",
                    "Setup will install the indicator, model, native trainer and browser studio.")
                : folders.Count > 0
                    ? T("Vyberte jeden terminál ze seznamu nebo otevřete jeho datovou složku.",
                        "Choose one terminal from the list or browse to its data folder.")
                    : T("V MT5 zvolte Soubor → Otevřít složku dat a tuto složku vyberte zde.",
                        "In MT5 select File → Open Data Folder, then choose that folder here.");
        }

        private void Browse()
        {
            using (var picker = new FolderBrowserDialog { ShowNewFolderButton = false,
                Description = T("Vyberte datovou složku z nabídky MT5 Soubor → Otevřít složku dat.",
                                "Select the folder opened by File → Open Data Folder in MT5.") })
            {
                if (path.Text.Length > 0) picker.SelectedPath = path.Text;
                if (picker.ShowDialog(this) != DialogResult.OK) return;
                try { terminals.SelectedIndex = AddTarget(Terminals.Explicit(picker.SelectedPath)); ShowSelection(); }
                catch (Exception)
                {
                    terminals.SelectedIndex = -1;
                    ShowSelection();
                    message.Text = T("Zvolená složka neobsahuje MQL5. Vyberte datovou složku svého terminálu.",
                                     "The selected folder does not contain MQL5. Choose your terminal's data folder.");
                }
            }
        }

        private async Task StartInstallation()
        {
            if (finished) { Close(); return; }
            if (installing || terminals.SelectedIndex < 0) return;
            string target = folders[terminals.SelectedIndex];
            installing = true;
            terminals.Enabled = browse.Enabled = primary.Enabled = false;
            progress.Visible = true;
            heading.Text = T("Instalace DNForecast", "Installing DNForecast");
            message.Text = T("Ověřuji a instaluji soubory do vybrané složky…", "Verifying and installing files in the selected folder…");
            IProgress<string> updates = new Progress<string>(s => message.Text = s);
            try { Result = await Task.Run(() => Program.Install(target, logPath, noWait, updates.Report, cancellation.Token)); }
            catch (OperationCanceledException) { Result = 1223; }
            catch (Exception e) { message.Text = e.Message; Result = 1; }
            installing = false;
            if (Result == 1223) { Close(); return; }
            progress.Visible = false;
            finished = true;
            cancel.Visible = false;
            studio.Visible = Result == 0;
            CancelButton = null;
            heading.Text = Result == 0 ? T("Instalace dokončena", "Installation complete") : T("Instalace se nedokončila", "Installation incomplete");
            if (Result == 0) message.Text = T("V MT5 obnovte seznam indikátorů a přidejte DNForecast_Async na graf.",
                                                 "Refresh Indicators in MT5 and attach DNForecast_Async to a chart.");
            primary.Text = T("Zavřít", "Close");
            primary.Enabled = true;
            primary.Focus();
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (disposing)
            {
                artwork.Dispose();
                cancellation.Dispose();
            }
        }
    }
}
