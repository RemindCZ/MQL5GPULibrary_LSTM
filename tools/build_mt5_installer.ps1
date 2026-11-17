param(
    [string]$ReleaseZip = (Join-Path (Split-Path $PSScriptRoot -Parent) 'output\forecast-release\DNForecast-3.1.0-win-x64.zip'),
    [string]$OutputDirectory = (Join-Path (Split-Path $PSScriptRoot -Parent) 'output\mt5-installer'),
    [string]$ExpectedZipSha256
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$archive = (Resolve-Path -LiteralPath $ReleaseZip).Path
if ($ExpectedZipSha256 -and (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $ExpectedZipSha256) {
    throw 'Release ZIP SHA-256 differs from the expected digest.'
}
$prefix = 'DNForecast-3.1.0-win-x64/'
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $output -Force | Out-Null
$exe = Join-Path $output 'DNForecast-3.1.0-MT5-Setup.exe'
$work = Join-Path $output ([guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$source = Join-Path $work 'Payload.cs'
$packed = Join-Path $work 'install.zip'
$intermediate = Join-Path $work 'DNForecast-3.1.0-MT5-Setup.exe'
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    $payload = [IO.Compression.ZipFile]::Open($packed, [IO.Compression.ZipArchiveMode]::Create)
    try {
        $reader = [IO.StreamReader]::new($zip.GetEntry($prefix+'SHA256.json').Open())
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        $items = foreach ($entry in $manifest) {
            $path = $entry.path
            if ($path.StartsWith('MQL5/')) { $target = $path }
            elseif ($path -eq 'docs/DNForecast_Manual_cs.pdf') { $target = 'MQL5/Files/DNForecast/DNForecast_Manual_cs.pdf' }
            elseif ($path -eq 'LICENSE.txt' -or $path.StartsWith('licenses/')) { $target = "MQL5/Files/DNForecast/$path" }
            else { continue }
            if ($path.Split('/') -contains '..' -or $path -notmatch '^[a-zA-Z0-9_./-]+$' -or $entry.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or -not $zip.GetEntry($prefix+$path)) {
                throw "Invalid release entry: $path"
            }
            $destination = $payload.CreateEntry($path).Open()
            $original = $zip.GetEntry($prefix+$path).Open()
            try { $original.CopyTo($destination) } finally { $original.Dispose(); $destination.Dispose() }
            $argsText = if ($entry.runtime_version) { ', "' + ([version]$entry.runtime_version).ToString() + '"' } else { '' }
            'new Item("{0}", "{1}", "{2}"{3}),' -f $path, $target, $entry.sha256.ToLowerInvariant(), $argsText
        }
    } finally { $payload.Dispose(); $zip.Dispose() }
    $definition = @(
        'namespace Remind.DNForecast.Setup { internal static class Payload {'
        ('internal const string ArchiveHash = "{0}";' -f (Get-FileHash $packed -Algorithm SHA256).Hash.ToLowerInvariant())
        'internal static readonly Item[] Items = {'
        $items
        '}; } }'
    )
    [IO.File]::WriteAllLines($source, [string[]]$definition, [Text.Encoding]::UTF8)
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    $sources = @(Get-ChildItem -Path (Join-Path $root 'installer\*.cs') -File | Sort-Object Name | ForEach-Object { $_.FullName })
    $switches = @('/nologo', '/target:winexe', '/platform:x64', '/optimize+', '/debug-', '/utf8output', '/codepage:65001',
        '/r:System.Core.dll', '/r:System.IO.Compression.dll', '/r:System.Windows.Forms.dll', '/r:System.Drawing.dll',
        "/win32manifest:$(Join-Path $root 'installer\app.manifest')", "/win32icon:$(Join-Path $root 'installer\assets\dnforecast.ico')",
        "/resource:$(Join-Path $root 'installer\assets\setup-header.jpg'),setup-header.jpg", "/out:$intermediate", "/resource:$packed,release.zip")
    & $compiler @switches @sources $source
    if ($LASTEXITCODE -ne 0) { throw "C# compiler failed with exit code $LASTEXITCODE" }
    Copy-Item -LiteralPath $intermediate -Destination $exe -Force
    Get-FileHash -LiteralPath $exe -Algorithm SHA256
} finally {
    foreach ($file in @($source, $packed, $intermediate)) { if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force } }
    [IO.Directory]::Delete($work)
}
