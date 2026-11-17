param(
    [Parameter(Mandatory=$true)][string]$OrtSdk,
    [string]$BuildDirectory='build\Release',
    [string]$OutputDirectory='output\forecast-release',
    [string]$CrtDirectory
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$build=Join-Path $root $BuildDirectory
$ort=(Resolve-Path -LiteralPath $OrtSdk).Path
$package=Join-Path (Join-Path $root $OutputDirectory) 'DNForecast-3.1.0-win-x64'
if ((Test-Path -LiteralPath $package) -or (Test-Path -LiteralPath "$package.zip")) {throw "Choose an unused output directory: $package"}
$map=[ordered]@{}
if (!$CrtDirectory) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) { throw 'Supply -CrtDirectory from the Visual Studio x64 redistributable.' }
    $CrtDirectory = Get-ChildItem -Directory (Join-Path $vs 'VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT') |
        Sort-Object { [version]$_.Parent.Parent.Name } -Descending | Select-Object -First 1 -ExpandProperty FullName
}
$crtFiles=@('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll')
foreach($dll in $crtFiles) { $map[(Join-Path $CrtDirectory $dll)]="MQL5\Libraries\$dll" }
foreach($file in @('DNForecast.dll','DNForecastTrain.dll','DNForecast-Train.exe','onnxruntime.dll','onnxruntime_providers_shared.dll')) {
    $map[(Join-Path $build $file)]="MQL5\Libraries\$file"
}
$map[(Join-Path $build 'DNForecast.lib')]='include\DNForecast.lib'
$map[(Join-Path $build 'DNForecastTrain.lib')]='include\DNForecastTrain.lib'
$map[(Join-Path $root 'docs\Native-Studio.cs.txt')]='MQL5\Files\DNForecast\Native-Studio.cs.txt'
foreach($name in @('LICENSE','ThirdPartyNotices.txt','VERSION_NUMBER')) {$map[(Join-Path $ort $name)]="licenses\onnxruntime\$name"}
foreach($file in @('LICENSE.txt','include\DNForecast.h','include\DNForecastTrain.h','MQL5\Include\DNForecast.mqh','docs\DNForecast.cs.md',
    'docs\DNForecast_Manual_cs.pdf','docs\research\ICTrading-2026-10.json',
    'tools\Install-MT5.ps1','examples\forecast.c','examples\train_native.cpp')) {
    $map[(Join-Path $root $file)]=$file
}
foreach($ext in @('mq5','ex5')) {
    foreach($stem in @('MQL5\Scripts\DNForecast_Smoke','MQL5\Scripts\DNForecast_Train','MQL5\Indicators\DNForecast_Async')) {
        $file="$stem.$ext"; $map[(Join-Path $root $file)]=$file
    }
}
foreach($ext in @('onnx','json')) {
    $file="AUDUSD-H1-TCN.$ext"; $map[(Join-Path $root "models\$file")]="MQL5\Files\DNForecast\$file"
}
$studio = Join-Path $build 'DNForecast-Studio.exe'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $compiler /nologo /target:winexe /platform:x64 /optimize+ /codepage:65001 /r:System.dll /r:System.Core.dll /r:System.Windows.Forms.dll /r:System.Drawing.dll /r:System.Web.dll /r:System.Web.Extensions.dll "/out:$studio" "/win32icon:$(Join-Path $root 'installer/assets/dnforecast.ico')" "/resource:$(Join-Path $root 'studio/index.html'),index.html" "/resource:$(Join-Path $root 'installer/assets/setup-header.jpg'),header.jpg" (Join-Path $root 'studio/Program.cs') (Join-Path $root 'studio/Server.cs')
if($LASTEXITCODE -ne 0){throw 'Studio compilation failed'}
$map[$studio]='MQL5\Libraries\DNForecast-Studio.exe'
foreach($source in $map.Keys) {if(-not (Test-Path -LiteralPath $source -PathType Leaf)){throw "Required file missing: $source"}}
if((Get-FileHash -LiteralPath (Join-Path $build 'onnxruntime.dll')).Hash -ne
   (Get-FileHash -LiteralPath (Join-Path $ort 'lib\onnxruntime.dll')).Hash) {throw 'Build and SDK do not match'}
foreach($source in $map.Keys) {
    $dest=Join-Path $package $map[$source]
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $dest
}
$manifest=foreach($dest in $map.Values) {
    $file=Join-Path $package $dest
    $row=[ordered]@{path=$dest.Replace('\','/');sha256=(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()}
    if ((Split-Path $dest -Leaf) -in $crtFiles) { $row.runtime_version=(Get-Item -LiteralPath $file).VersionInfo.FileVersion }
    [PSCustomObject]$row
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'SHA256.json') -Encoding UTF8
Compress-Archive -LiteralPath $package -DestinationPath "$package.zip"
Get-FileHash -LiteralPath "$package.zip"
