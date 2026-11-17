param(
    [Parameter(Mandatory=$true)][string]$Setup,
    [Parameter(Mandatory=$true)][string]$ReleaseZip,
    [switch]$InteractiveSmoke
)
$ErrorActionPreference = 'Stop'
$setupExe = (Resolve-Path -LiteralPath $Setup).Path
$archive = (Resolve-Path -LiteralPath $ReleaseZip).Path
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('DNForecast-Setup-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null
function Assert($value, $message) { if (!$value) { throw $message } }
function Folder($path) { [IO.Directory]::CreateDirectory($path) | Out-Null }
function Seed($path, $value) { Folder (Split-Path $path -Parent); [IO.File]::WriteAllText($path, $value) }
function Start-Test([string]$file, [string[]]$arguments) {
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.UseShellExecute = $false
    $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $start.FileName = $file
    $start.Arguments = (($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' ')
    $start.WorkingDirectory = $scratch
    return [Diagnostics.Process]::Start($start)
}
function Result($p, $expected) {
    try {
        if (!$p.WaitForExit(45000)) { $p.Kill(); $p.WaitForExit(); throw 'Installer timed out.' }
        Assert ($p.ExitCode -eq $expected) "Exit $($p.ExitCode), expected $expected. See $scratch"
    } finally { $p.Dispose() }
}
function Quiet([string]$root) {
    $p = Start-Test $setupExe @('/quiet','/no-wait','/target',$root,'/log',(Join-Path $scratch 'test.log'))
    Result $p 0
}
try {
    $root = Join-Path $scratch 'Obchodní terminál 5'
    Folder (Join-Path $root 'MQL5')
    Seed (Join-Path $root 'MQL5\Indicators\Other.mq5') 'unrelated'
    Quiet $root
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        $reader = [IO.StreamReader]::new($zip.GetEntry('DNForecast-3.0.0-win-x64/SHA256.json').Open())
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        $items = @($manifest | Where-Object { $_.path -like 'MQL5/*' })
        Assert ($items.Count -eq 10) 'Unexpected release layout.'
        foreach ($item in $items) {
            Assert ((Get-FileHash -LiteralPath (Join-Path $root $item.path)).Hash -eq $item.sha256) "Hash: $($item.path)"
        }
        foreach ($item in $manifest | Where-Object { $_.path -like 'licenses/*' -or $_.path -eq 'LICENSE.txt' -or $_.path -like '*.pdf' }) {
            $relative = if ($item.path -like '*.pdf') { 'DNForecast_Manual_cs.pdf' } else { $item.path }
            $path = Join-Path $root "MQL5/Files/DNForecast/$relative"
            Assert ((Get-FileHash -LiteralPath $path).Hash -eq $item.sha256) "Manual/license: $relative"
        }
    } finally { $zip.Dispose() }

    $dll = Join-Path $root 'MQL5\Libraries\DNForecast.dll'
    $original = (Get-FileHash -LiteralPath $dll).Hash
    $mtime = (Get-Item -LiteralPath $dll).LastWriteTimeUtc
    Quiet $root
    Assert ((Get-Item -LiteralPath $dll).LastWriteTimeUtc -eq $mtime) 'Repeat install changed existing files.'
    Assert (!(Test-Path -LiteralPath (Join-Path $root 'DNForecast.Backups'))) 'Backup created on clean install.'
    Seed $dll 'old DLL'
    $hold = [IO.File]::Open($dll, 'Open', 'ReadWrite', 'None')
    try {
        Result (Start-Test $setupExe @('/quiet','/no-wait','/target',$root,'/log',"$scratch\locked.log")) 32
        $waiting = Start-Test $setupExe @('/quiet','/target',$root,'/log',"$scratch\waiting.log")
        $until = [DateTime]::UtcNow.AddSeconds(10)
        while ([DateTime]::UtcNow -lt $until) {
            if ((Test-Path "$scratch\waiting.log") -and (Select-String -LiteralPath "$scratch\waiting.log" -Pattern 'Waiting for file' -Quiet)) { break }
            Start-Sleep -Milliseconds 100
        }
        Assert (!$waiting.HasExited -and (Select-String -LiteralPath "$scratch\waiting.log" -Pattern 'Waiting for file' -Quiet)) 'Setup did not wait for a locked DLL.'
    } finally { $hold.Dispose() }
    Result $waiting 0
    Assert ((Get-FileHash -LiteralPath $dll).Hash -eq $original) 'Update failed after unlock.'
    $backup = @(Get-ChildItem -LiteralPath (Join-Path $root 'DNForecast.Backups') -Filter DNForecast.dll -Recurse -File)
    Assert ($backup.Count -eq 1 -and [IO.File]::ReadAllText($backup[0].FullName) -eq 'old DLL') 'Previous DLL backup missing.'

    # Simulate a process crash just after one atomic replacement. A directory at a later
    # file destination also forces rollback of the subsequent attempted installation.
    $broken = Join-Path $scratch 'Interrupted'
    $work = Join-Path $broken '.DNForecast-Setup-v1'
    $saved = Join-Path $work 'old/MQL5/Libraries/DNForecast.dll'
    Seed $saved 'prior version'
    $prior = (Get-FileHash $saved).Hash.ToLowerInvariant()
    $live = Join-Path $broken 'MQL5/Libraries/DNForecast.dll'
    Folder (Split-Path $live -Parent)
    Copy-Item -LiteralPath $dll -Destination $live
    $tab = [char]9
    Seed (Join-Path $work 'prepared') ("MQL5/Libraries/DNForecast.dll" + $tab + $prior + $tab + $original.ToLowerInvariant())
    Folder (Join-Path $broken 'MQL5/Files/DNForecast/AUDUSD-H1-TCN.onnx')
    Result (Start-Test $setupExe @('/quiet','/no-wait','/target',$broken,'/log',"$scratch\rollback.log")) 1
    Assert ([IO.File]::ReadAllText($live) -eq 'prior version') 'Failed transaction left partial update.'
    Assert (!(Test-Path (Join-Path $broken 'MQL5/Include/DNForecast.mqh'))) 'New file survived rollback.'
    Assert (!(Test-Path $work)) 'Journal not cleaned.'
    Assert ([IO.File]::ReadAllText((Join-Path $root 'MQL5/Indicators/Other.mq5')) -eq 'unrelated') 'Unrelated file changed.'

    # Exercise automatic discovery with isolated registrations. No real terminal is modified.
    $asm = [Reflection.Assembly]::Load([IO.File]::ReadAllBytes($setupExe))
    $flags = [Reflection.BindingFlags]'Static,NonPublic'
    $discover = $asm.GetType('Remind.DNForecast.Setup.Terminals').GetMethod('Discover',$flags)
    $roaming = Join-Path $scratch 'Roaming'
    $standard = Join-Path $roaming ('MetaQuotes\Terminal\' + ('1' * 32))
    $programs = Join-Path $scratch 'Program Files'
    $broker = Join-Path $programs 'Broker'
    Seed (Join-Path $broker 'terminal64.exe') 'test registration'
    Folder (Join-Path $broker 'MQL5/Logs')
    Folder (Join-Path $standard 'MQL5')
    [IO.File]::WriteAllText((Join-Path $standard 'origin.txt'),$broker,[Text.Encoding]::Unicode)
    $portable = Join-Path $scratch 'Portable'
    Seed (Join-Path $portable 'terminal64.exe') 'test registration'
    Folder (Join-Path $portable 'MQL5/Logs')
    $portableProfile = Join-Path $roaming ('MetaQuotes\Terminal\' + ('4' * 32))
    Seed (Join-Path $portableProfile 'origin.txt') $portable
    $customBroker = Join-Path $scratch 'Installed broker on another drive'
    Seed (Join-Path $customBroker 'terminal64.exe') 'registered installation'
    $customProfile = Join-Path $roaming ('MetaQuotes\Terminal\' + ('5' * 32))
    Seed (Join-Path $customProfile 'origin.txt') $customBroker
    Folder (Join-Path $customProfile 'MQL5')
    $lookalike = Join-Path $scratch 'Program Files-copy\Broker'
    Seed (Join-Path $lookalike 'terminal64.exe') 'unregistered copy'
    $lookalikeProfile = Join-Path $roaming ('MetaQuotes\Terminal\' + ('6' * 32))
    Seed (Join-Path $lookalikeProfile 'origin.txt') $lookalike
    Folder (Join-Path $lookalikeProfile 'MQL5')
    $stale = Join-Path $roaming ('MetaQuotes\Terminal\' + ('2' * 32))
    Seed (Join-Path $stale 'origin.txt') (Join-Path $scratch 'Removed broker')
    Folder (Join-Path $stale 'MQL5')
    $mt4 = Join-Path $roaming ('MetaQuotes\Terminal\' + ('3' * 32))
    Seed (Join-Path $mt4 'origin.txt') "$scratch\MT4"
    Seed "$scratch\MT4\terminal.exe" 'MT4'
    Folder "$mt4\MQL4"
    $registered = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [void]$registered.Add($customBroker)
    $argsDiscover = [object[]]@($roaming.PSObject.BaseObject,[string[]]@($programs),$registered.PSObject.BaseObject,[Action[string]]{param($s)})
    $found = $discover.Invoke($null,$argsDiscover)
    Assert ($found.Count -eq 2 -and $found.Contains($standard) -and $found.Contains($customProfile)) 'Expected only installed brokers data profiles.'
    Folder (Join-Path $portableProfile 'MQL5')
    $found = $discover.Invoke($null,$argsDiscover)
    Assert ($found.Count -eq 2 -and !$found.Contains($portableProfile) -and !$found.Contains($portable)) 'A portable registration with MQL5 became a target.'
    $argsDiscover[0] = [IO.Path]::Combine($scratch,'Uninitialized')
    Assert ($discover.Invoke($null,$argsDiscover).Count -eq 0) 'Missing MT5 should be reported.'

    # On a clean Windows host the optional VC runtime is deployed app-locally.
    $missingSystem = Join-Path $scratch 'Empty system directory'
    $class = $asm.GetType('Remind.DNForecast.Setup.Installation')
    $payload = $class.GetMethod('OpenPayload',$flags).Invoke($null,$null)
    try {
        [void]$class.GetMethod('Apply',$flags).Invoke($null,[object[]]@($portable.PSObject.BaseObject,$payload.PSObject.BaseObject,[Action[string]]{param($s)},$missingSystem.PSObject.BaseObject))
    } finally { $payload.Dispose() }
    foreach ($name in @('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll')) {
        Assert (Test-Path (Join-Path $portable "MQL5/Libraries/$name")) "Missing app-local $name"
    }

    if ($InteractiveSmoke) {
        $desktopShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
        & $desktopShell -NoProfile -STA -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'test_mt5_setup_ui.ps1') -Setup $setupExe -Scratch $scratch
        if ($LASTEXITCODE -ne 0) { throw 'Installer window tests failed.' }
    }
    Write-Output 'PASS: package hashes, discovery, VC fallback, idempotence, locked update, backup, crash recovery and rollback.'
} finally {
    $testRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
    $resolved = [IO.Path]::GetFullPath($scratch)
    if ((Split-Path $resolved -Parent) -ne $testRoot -or (Split-Path $resolved -Leaf) -notmatch '^DNForecast-Setup-[0-9a-f]{32}$') { throw 'Unexpected test directory' }
    Remove-Item -LiteralPath $resolved -Force -Recurse
}
