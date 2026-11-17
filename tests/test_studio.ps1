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
    $report=Invoke-RestMethod "$url/report/$id"
    $sourceRows=@($data.Trim() -split '\r?\n' | Select-Object -Skip 1)
    $first=[int][Math]::Floor($sourceRows.Count*.85)
    if($report.points.Count -ne $sourceRows.Count-$first -or $report.horizon -ne 6){throw 'Unexpected holdout boundary'}
    $modelSSE=0.0;$baseSSE=0.0;$covered=0
    for($i=0;$i -lt $report.points.Count;$i++){
        $point=$report.points[$i]
        if($point[0] -ne [long]($sourceRows[$first+$i].Split(',')[0])){throw 'Forecast target time differs'}
        if($i -lt 6){if($null -ne $point[2]){throw 'Forecast plotted before maturity'};continue}
        $anchor=$report.points[$i-6][1]
        $actual=[Math]::Log($point[1]/$anchor)
        $prediction=[Math]::Log($point[2]/$anchor)
        $baseSSE+=$actual*$actual
        $modelSSE+=[Math]::Pow($prediction-$actual,2)
        if($point[1] -ge $point[3] -and $point[1] -le $point[4]){$covered++}
    }
    $n=$report.points.Count-6
    $lastScore=($job.log.Trim() -split '\r?\n')[-1].Split(',')
    if([Math]::Abs($modelSSE/$n-[double]$lastScore[1]) -gt [Math]::Max(1e-12,$modelSSE/$n*.0001) -or
       [Math]::Abs($baseSSE/$n-[double]$lastScore[2]) -gt [Math]::Max(1e-12,$baseSSE/$n*.0001) -or
       [Math]::Abs($covered/$n-[double]$lastScore[3]) -gt .00001){throw 'Trace differs from native holdout scores'}
    $prices=@{};$index=@{}
    for($i=0;$i -lt $report.points.Count;$i++){$prices[[long]$report.points[$i][0]]=$report.points[$i][1];$index[[long]$report.points[$i][0]]=$i}
    $equity=1.0;$previous=0
    foreach($trade in $report.trades) {
        if($index[[long]$trade.entry] -ne $index[[long]$trade.signal]+1 -or $index[[long]$trade.exit] -ne $index[[long]$trade.entry]+6 -or $trade.entry -lt $previous){throw 'Trade timing or overlap'}
        $ratio=$prices[[long]$trade.exit]/$prices[[long]$trade.entry]
        $sign=if($trade.side -eq 'long'){1}else{-1}
        $expected=$sign*($ratio-1)-.0001*(1+$ratio)
        if([Math]::Abs($trade.gain-$expected) -gt .00000001){throw 'Round trip cost/return mismatch'}
        $equity*=1+$expected;$previous=$trade.exit
    }
    if([Math]::Abs($report.net-($equity-1)) -gt .00000001){throw 'Equity accounting mismatch'}
    $download=Join-Path $work 'verified.onnx'
    Invoke-WebRequest "$url/model/$id" -UseBasicParsing -OutFile $download
    if((Get-FileHash $download).Hash -ne (Get-FileHash (Join-Path $work $job.model)).Hash){throw 'Download mismatch'}
    Stop-TestStudio $p
    $p=Start-Studio
    $restored=(Invoke-RestMethod "$url/api/state").jobs | Where-Object id -eq $id
    if($restored.model -ne $job.model){throw 'Completed job did not survive a server restart'}
    # Exercise the file protocol and restart a persistent plan with the same snapshot.
    $page=(Invoke-WebRequest "$url/" -UseBasicParsing).Content
    $header=@{'X-DNForecast-Key'=[regex]::Match($page,"token='([A-F0-9]{64})'").Groups[1].Value}
    $bridge=Join-Path $work 'Bridge'
    $recipe='target=BTCUSD&drivers=&seconds=3600&bars=720&steps=96&horizons=6&epochs=2&minutes=1'
    function Feed { [IO.File]::WriteAllLines((Join-Path $bridge 'status.txt'),@('DNBRIDGE1','online','Demo','BTCUSD')) }
    function Answer {
        for($t=0;$t -lt 80;$t++){
            $req=Get-ChildItem -LiteralPath $bridge -Filter '*.req' | Select-Object -First 1
            if($req){break};Start-Sleep -Milliseconds 150
        }
        if(!$req -or [IO.File]::ReadAllText($req.FullName) -ne '3600|720|BTCUSD'){throw 'Invalid bridge request'}
        $out=[IO.Path]::ChangeExtension($req.FullName,'.csv')
        [IO.File]::WriteAllText($out+'.tmp',$snapshot);[IO.File]::Move($out+'.tmp',$out)
        for($t=0;$t -lt 100;$t++){
            $state=Invoke-RestMethod "$url/api/state";if(!$state.busy){return $state}
            Start-Sleep -Milliseconds 150
        }
        throw 'Scheduled job timed out'
    }
    try{Invoke-WebRequest "$url/api/run?$recipe" -UseBasicParsing -Headers $header -Method Post -Body '' | Out-Null;throw 'Offline MT5 accepted'}catch{if([int]$_.Exception.Response.StatusCode -ne 409){throw}}
    $snapshot="close_time,available_at,BTCUSD"+[Environment]::NewLine+(($sourceRows | Select-Object -Last 720 | ForEach-Object {($_.Split(',')[0..2]) -join ','}) -join [Environment]::NewLine)
    Feed
    $recipe+='&engine=cpu&batch=16&rate=0.002'
    $graph="RNN 4 linear 0 concat -1`nDense 4 tanh 0 concat 0`nIdentity 0 linear 0 sum 0 1"
    try {Invoke-WebRequest "$url/api/plan?$recipe" -UseBasicParsing -Headers $header -Method Post -Body "Dense 4 tanh 0 concat -1`nIdentity 0 linear 0 sum 0 -1" | Out-Null;throw 'Invalid residual dimensions accepted'}catch{if([int]$_.Exception.Response.StatusCode -ne 400){throw}}
    Invoke-WebRequest "$url/api/plan?$recipe" -UseBasicParsing -Headers $header -Method Post -Body $graph | Out-Null
    $scheduled=Answer
    if(!$scheduled.jobs[0].hasReport -or !$scheduled.plan.fingerprint -or $scheduled.jobs[0].config.graph -ne $graph){throw 'Scheduled graph training failed'}
    Stop-TestStudio $p
    $planFile=Join-Path $work 'studio-plan.json'
    $plan=[IO.File]::ReadAllText($planFile) | ConvertFrom-Json
    $plan.next=[DateTime]::UtcNow.AddMinutes(-1).ToString('o')
    [IO.File]::WriteAllText($planFile,($plan|ConvertTo-Json -Depth 5))
    Feed
    $p=Start-Studio
    $again=Answer
    if($again.plan.recipe.graph -ne $graph -or $again.plan.recipe.engine -ne 'cpu'){throw 'Saved topology or backend changed after restart'}
    if($again.jobs[0].model -or $again.plan.fingerprint -ne $scheduled.plan.fingerprint -or $again.jobs[0].status -eq 'Chyba'){throw 'Unchanged data triggered another model'}
    $page=(Invoke-WebRequest "$url/" -UseBasicParsing).Content
    $header=@{'X-DNForecast-Key'=[regex]::Match($page,"token='([A-F0-9]{64})'").Groups[1].Value}
    Invoke-WebRequest "$url/api/stop-plan" -UseBasicParsing -Headers $header -Method Post -Body '' | Out-Null
    if((Invoke-RestMethod "$url/api/state").plan -or (Test-Path $planFile)){throw 'Plan still active'}
    Write-Output "PASS: scheduled MT5 acquisition, unchanged snapshot, replay, accounting, import, background native training, model download and persisted results. $($job.contract)"
} finally {
    Stop-TestStudio $p
    $dir=[IO.Path]::GetFullPath($work)
    if ((Split-Path $dir -Parent) -ne [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') -or (Split-Path $dir -Leaf) -notmatch '^DNStudio-[0-9a-f]{32}$'){throw 'Unexpected scratch directory'}
    Remove-Item -LiteralPath $dir -Recurse -Force
}
