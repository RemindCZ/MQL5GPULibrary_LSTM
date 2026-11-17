param([Parameter(Mandatory=$true)][string]$Trainer,[switch]$Cuda)
$ErrorActionPreference='Stop'
$exe=(Resolve-Path $Trainer).Path
$work=Join-Path ([IO.Path]::GetTempPath()) ('DNGraph-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($work)
try{
    $culture=[Globalization.CultureInfo]::InvariantCulture
    $rows=@('close_time,available_at,TEST,AUX')
    for($i=0;$i -lt 640;$i++){
        $t=1600000000+3600*$i
        $target=(100+[Math]::Sin(.16*$i)).ToString('R',$culture)
        $aux=if($i%19 -gt 2){(44+[Math]::Cos(.09*$i)).ToString('R',$culture)}else{''}
        $rows+="$t,$t,$target,$aux"
    }
    $csv=Join-Path $work 'prices.csv'
    [IO.File]::WriteAllLines($csv,[string[]]$rows)
    $cfg=Join-Path $work 'graph.dng'
    $modes=@('cpu')
    if($Cuda){& $exe --cuda-device;if($LASTEXITCODE -ne 0){throw 'CUDA probe failed'};$modes+='cuda'}
    foreach($mode in $modes){
        $model=Join-Path $work "$mode.onnx";$trace=Join-Path $work "$mode.csv"
        [IO.File]::WriteAllLines($cfg,@("$mode 32 .003",'LSTM 7 linear 0.1 concat -1','GRU 7 linear 0 concat -1','Identity 0 linear 0 sum 0 1','RNN 4 linear 0 concat 2 -1','Dense 5 relu 0 concat 3'))
        $metrics=& $exe $csv $model 3600 16 2 2 $trace $cfg
        if($LASTEXITCODE -ne 0 -or !(Test-Path $model) -or $metrics.Count -ne 3){throw "$mode fit or production ONNX parity failed"}
        $points=@(Import-Csv $trace)
        if($points.Count -ne 96 -or $points[-1].point -ne '' -or $points[0].q10 -eq '' -or [double]$points[0].q10 -gt [double]$points[0].q90){throw 'Invalid holdout trace'}
        Write-Output "$mode graph trained and verified with missing auxiliary data, skip connections and all five operators."
    }
    [IO.File]::WriteAllLines($cfg,@('cpu 16 .001','Dense 7 tanh 0 concat -1','Dense 5 tanh 0 sum -1 0'))
    & $exe $csv (Join-Path $work 'bad.onnx') 3600 16 2 1 (Join-Path $work 'bad.csv') $cfg
    if($LASTEXITCODE -eq 0 -or (Test-Path (Join-Path $work 'bad.onnx'))){throw 'Invalid sum accepted'}
} finally {
    $root=[IO.Path]::GetFullPath($work)
    if((Split-Path $root -Parent) -ne [IO.Path]::GetTempPath().TrimEnd('\') -or (Split-Path $root -Leaf) -notmatch '^DNGraph-[0-9a-f]{32}$'){throw 'Unexpected test path'}
    Remove-Item -LiteralPath $root -Recurse -Force
}
exit 0
