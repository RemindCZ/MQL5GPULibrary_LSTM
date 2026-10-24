param(
    [Parameter(Mandatory=$true)][string]$TerminalDataPath,
    [string]$PackageRoot = (Split-Path $PSScriptRoot -Parent),
    [string]$CudaPath = $env:CUDA_PATH,
    [switch]$CopyCudaRuntime,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$target = [IO.Path]::GetFullPath($TerminalDataPath)
if (-not (Test-Path -LiteralPath (Join-Path $target 'MQL5') -PathType Container)) {
    throw 'Choose the terminal DATA folder shown by MT5: File > Open Data Folder (must contain MQL5).'
}
$mql = Join-Path ([IO.Path]::GetFullPath($PackageRoot)) 'MQL5'
if (-not (Test-Path -LiteralPath (Join-Path $mql 'Libraries\MQL5GPULibrary_LSTM.dll'))) {
    throw 'Use this installer from the built release package, not the source checkout.'
}
$files = @(Get-ChildItem -LiteralPath $mql -File -Recurse)
$copies = @()
foreach ($file in $files) {
    $relative = $file.FullName.Substring($mql.Length).TrimStart('\','/')
    $copies += @{Source=$file.FullName; Destination=(Join-Path (Join-Path $target 'MQL5') $relative)}
}
if ($CopyCudaRuntime) {
    if (-not $CudaPath) { throw 'Set -CudaPath or install the matching CUDA Toolkit.' }
    foreach ($name in @('cublas64_13.dll','cublasLt64_13.dll','curand64_10.dll')) {
        $found = @((Join-Path $CudaPath "bin\x64\$name"),(Join-Path $CudaPath "bin\$name")) |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        if (-not $found) { throw "Missing CUDA 13 runtime: $name" }
        $copies += @{Source=$found; Destination=(Join-Path $target "MQL5\Libraries\$name")}
    }
}
foreach ($copy in $copies) {
    if ((Test-Path -LiteralPath $copy.Destination) -and -not $Force) {
        throw "Already exists: $($copy.Destination). Close MT5 and use -Force only to replace this package's files."
    }
}
foreach ($copy in $copies) {
    New-Item -ItemType Directory -Path (Split-Path $copy.Destination -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $copy.Source -Destination $copy.Destination -Force:$Force
}
Write-Output "Installed $($copies.Count) files into $target. Restart MT5, enable DLL imports and run LSTM_SmokeTest."
Write-Output 'An NVIDIA driver/GPU and Microsoft Visual C++ 2015-2022 x64 runtime are still required.'
