param([Parameter(Mandatory=$true)][string]$Studio,[string]$Csv)
$ErrorActionPreference='Stop'
$exe=(Resolve-Path -LiteralPath $Studio).Path
$work=Join-Path ([IO.Path]::GetTempPath()) ('DNStudio-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($work)
$probe=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
$probe.Start()
$port=$probe.LocalEndpoint.Port
$probe.Stop()
$url="http://127.0.0.1:$port"
function Start-Studio {
    $p=Start-Process -FilePath $exe -ArgumentList ('--headless --port '+$port+' --data "'+$work+'"') -WindowStyle Hidden -PassThru
    try {
        for($i=0;$i -lt 100;$i++) {
            if($p.HasExited){throw "Studio exited with $($p.ExitCode)"}
            try { $null=Invoke-RestMethod -Uri "$url/api/state" -TimeoutSec 2;return $p }catch{Start-Sleep -Milliseconds 100}
        }
        throw 'Studio startup timeout'
    }catch{if(!$p.HasExited){$p.Kill();$p.WaitForExit()};throw}
}
function Stop-TestStudio($p) {if($p -and !$p.HasExited){if($p.Path -ne $exe){throw 'Unexpected process'};$p.Kill();$p.WaitForExit()}}
$p=$null
try {
    $p=Start-Studio
    $page=(Invoke-WebRequest "$url/" -UseBasicParsing).Content
    $key=[regex]::Match($page,"token='([A-F0-9]{64})'").Groups[1].Value
    if(!$key){throw 'Studio access key missing'}
    $header=@{'X-DNForecast-Key'=$key}
    if($Csv){$data=[IO.File]::ReadAllText((Resolve-Path $Csv).Path)}
    else {
        $rows=@('close_time,available_at,BTCUSD')
        for($i=0;$i -lt 720;$i++){
            $time=1600000000+3600*$i
            $value=(30000+120*[Math]::Sin($i*.12)).ToString('R',[Globalization.CultureInfo]::InvariantCulture)
            $rows+="$time,$time,$value"
        }
        $data=$rows -join [Environment]::NewLine
    }
    $name=(Invoke-WebRequest "$url/api/import?name=BTCUSD-3600s-test.csv" -UseBasicParsing -Headers $header -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($data)) -ContentType 'application/octet-stream').Content
    $request="$url/api/train?file=$name&seconds=3600&steps=96&horizons=6&epochs=2"
    try {Invoke-WebRequest $request -UseBasicParsing -Method Post -Body '' | Out-Null;throw 'Missing key was accepted'}catch{if([int]$_.Exception.Response.StatusCode -ne 403){throw}}
    $id=(Invoke-WebRequest $request -UseBasicParsing -Headers $header -Method Post -Body '').Content
    Start-Sleep -Seconds 2
    for($i=0;$i -lt 200;$i++){
        $state=Invoke-RestMethod "$url/api/state"
        if(!$state.busy){break}
        Start-Sleep -Milliseconds 150
    }
    $job=$state.jobs | Where-Object id -eq $id
    if(!$job.model -or !$job.log.Contains('last_price_mse')) {throw "Training failed: $($job.log)"}
    $download=Join-Path $work 'verified.onnx'
    Invoke-WebRequest "$url/model/$id" -UseBasicParsing -OutFile $download
    if((Get-FileHash $download).Hash -ne (Get-FileHash (Join-Path $work $job.model)).Hash){throw 'Download mismatch'}
    Stop-TestStudio $p
    $p=Start-Studio
    $restored=(Invoke-RestMethod "$url/api/state").jobs | Where-Object id -eq $id
    if($restored.model -ne $job.model){throw 'Completed job did not survive a server restart'}
    Write-Output "PASS: import, background native training, model download and persisted results. $($job.contract)"
} finally {
    Stop-TestStudio $p
    $dir=[IO.Path]::GetFullPath($work)
    if ((Split-Path $dir -Parent) -ne [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') -or (Split-Path $dir -Leaf) -notmatch '^DNStudio-[0-9a-f]{32}$'){throw 'Unexpected scratch directory'}
    Remove-Item -LiteralPath $dir -Recurse -Force
}
