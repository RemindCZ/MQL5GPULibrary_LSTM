param(
    [string]$MetaEditor = 'C:\Program Files\IC Trading (MU) MT5 Terminal\MetaEditor64.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path -LiteralPath $MetaEditor)) { throw "Set -MetaEditor to your MetaEditor64.exe path" }
$sources = @('MQL5\Indicators\LSTM_CrossAsset_Forecast.mq5','MQL5\Scripts\LSTM_SmokeTest.mq5')
foreach ($source in $sources) {
    $file = Join-Path $projectRoot $source
    $started = Get-Date
    $arguments = @('/compile:"' + $file + '"','/include:"' + (Join-Path $projectRoot 'MQL5') + '"','/log')
    $job = Start-Process -FilePath $MetaEditor -ArgumentList $arguments -WindowStyle Hidden -PassThru
    if (-not $job.WaitForExit(60000)) { throw "MetaEditor timed out compiling $file" }
    $log = [IO.Path]::ChangeExtension($file,'.log')
    $text = Get-Content -LiteralPath $log -Raw
    $result = [regex]::Match($text,'Result:.*').Value
    Write-Output "$source : $result"
    $binary = [IO.Path]::ChangeExtension($file,'.ex5')
    if ($text -notmatch 'Result: 0 errors, 0 warnings' -or -not (Test-Path -LiteralPath $binary) -or
        (Get-Item -LiteralPath $log).LastWriteTime -lt $started.AddSeconds(-2) -or
        (Get-Item -LiteralPath $binary).LastWriteTime -lt $started.AddSeconds(-2)) {
        throw "MetaEditor verification failed. Read $log"
    }
}
