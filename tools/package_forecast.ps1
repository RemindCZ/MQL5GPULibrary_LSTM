param(
    [Parameter(Mandatory=$true)][string]$OrtSdk,
    [string]$BuildDirectory='build\Release',
    [string]$OutputDirectory='output\forecast-release'
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$build=Join-Path $root $BuildDirectory
$ort=(Resolve-Path -LiteralPath $OrtSdk).Path
$package=Join-Path (Join-Path $root $OutputDirectory) 'DNForecast-3.0.0-win-x64'
if ((Test-Path -LiteralPath $package) -or (Test-Path -LiteralPath "$package.zip")) {throw "Choose an unused output directory: $package"}
$map=[ordered]@{}
foreach($file in @('DNForecast.dll','onnxruntime.dll','onnxruntime_providers_shared.dll')) {
    $map[(Join-Path $build $file)]="MQL5\Libraries\$file"
}
$map[(Join-Path $build 'DNForecast.lib')]='include\DNForecast.lib'
foreach($name in @('LICENSE','ThirdPartyNotices.txt','VERSION_NUMBER')) {$map[(Join-Path $ort $name)]="licenses\onnxruntime\$name"}
foreach($file in @('LICENSE.txt','include\DNForecast.h','MQL5\Include\DNForecast.mqh','docs\DNForecast.cs.md',
    'docs\DNForecast_Manual_cs.pdf','docs\research\ICTrading-2026-10.json',
    'python\dnforecast.py','python\forecast_series.py','tools\Install-MT5.ps1','tools\bench_forecast.py',
    'examples\forecast.c','examples\replay_forecast.py','examples\train_forecast.py',
    'examples\export_mt5_history.py','examples\live_forecast.py','examples\backtest_strategy.py','examples\pack_legacy.py','examples\select_tasks.py')) {
    $map[(Join-Path $root $file)]=$file
}
foreach($ext in @('mq5','ex5')) {
    foreach($stem in @('MQL5\Scripts\DNForecast_Smoke','MQL5\Indicators\DNForecast_Async')) {
        $file="$stem.$ext"; $map[(Join-Path $root $file)]=$file
    }
}
foreach($ext in @('onnx','json')) {
    $file="AUDUSD-H1-TCN.$ext"; $map[(Join-Path $root "models\$file")]="MQL5\Files\DNForecast\$file"
}
foreach($source in $map.Keys) {if(-not (Test-Path -LiteralPath $source -PathType Leaf)){throw "Required file missing: $source"}}
if((Get-FileHash -LiteralPath (Join-Path $build 'onnxruntime.dll')).Hash -ne
   (Get-FileHash -LiteralPath (Join-Path $ort 'lib\onnxruntime.dll')).Hash) {throw 'Build and SDK do not match'}
foreach($source in $map.Keys) {
    $dest=Join-Path $package $map[$source]
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $dest
}
$manifest=foreach($dest in $map.Values) {
    [PSCustomObject]@{path=$dest.Replace('\','/');sha256=(Get-FileHash -LiteralPath (Join-Path $package $dest)).Hash.ToLowerInvariant()}
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'SHA256.json') -Encoding UTF8
Compress-Archive -LiteralPath $package -DestinationPath "$package.zip"
Get-FileHash -LiteralPath "$package.zip"
