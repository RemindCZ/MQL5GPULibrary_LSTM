param([string]$OutputDirectory = 'output\release')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$cache=Get-Content -LiteralPath (Join-Path $root 'build\CMakeCache.txt') -Raw
$architecture=[regex]::Match($cache,'(?m)^CMAKE_CUDA_ARCHITECTURES:[^=]+=([^\r\n]+)').Groups[1].Value
$cudaVersion=[regex]::Match($cache,'\[v(13\.[0-9.]+)\(').Groups[1].Value
if(-not $architecture -or -not $cudaVersion) { throw 'This package installer targets a CMake CUDA 13 build; check the build cache.' }
$output=[IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
$package=Join-Path $output 'MQL5GPULibrary-2.3.0'
if (Test-Path -LiteralPath $package) { throw "Package already exists: $package. Choose another -OutputDirectory." }
$required=@('build\Release\MQL5GPULibrary_LSTM.dll','MQL5\Indicators\LSTM_CrossAsset_Forecast.ex5',
    'MQL5\Scripts\LSTM_SmokeTest.ex5','output\pdf\MQL5GPULibrary_Manual_cs.pdf')
foreach($file in $required) { if(-not (Test-Path -LiteralPath (Join-Path $root $file))) { throw "Build first: $file" } }
foreach($dir in @('MQL5\Libraries','MQL5\Include','MQL5\Indicators','MQL5\Scripts','docs','python','examples','tools','include')) {
    New-Item -ItemType Directory -Path (Join-Path $package $dir) -Force | Out-Null
}
Copy-Item -LiteralPath (Join-Path $root $required[0]) -Destination (Join-Path $package 'MQL5\Libraries')
foreach($name in @('MQL5GPULibrary.mqh','MQL5GPUTopology.mqh')) {
    Copy-Item -LiteralPath (Join-Path $root "MQL5\Include\$name") -Destination (Join-Path $package 'MQL5\Include')
}
foreach($extension in @('mq5','ex5')) {
    Copy-Item -LiteralPath (Join-Path $root "MQL5\Indicators\LSTM_CrossAsset_Forecast.$extension") -Destination (Join-Path $package 'MQL5\Indicators')
    Copy-Item -LiteralPath (Join-Path $root "MQL5\Scripts\LSTM_SmokeTest.$extension") -Destination (Join-Path $package 'MQL5\Scripts')
}
Copy-Item -Path (Join-Path $root 'python\*.py') -Destination (Join-Path $package 'python')
Copy-Item -Path (Join-Path $root 'examples\*.py') -Destination (Join-Path $package 'examples')
Copy-Item -Path (Join-Path $root 'docs\*.md') -Destination (Join-Path $package 'docs')
Copy-Item -LiteralPath (Join-Path $root $required[3]) -Destination (Join-Path $package 'docs')
Copy-Item -LiteralPath (Join-Path $root 'include\MQL5GPULibrary.h') -Destination (Join-Path $package 'include')
Copy-Item -LiteralPath (Join-Path $root 'build\Release\MQL5GPULibrary_LSTM.lib') -Destination (Join-Path $package 'include')
Copy-Item -LiteralPath (Join-Path $root 'tools\Install-MT5.ps1') -Destination (Join-Path $package 'tools')
Copy-Item -LiteralPath (Join-Path $root 'LICENSE.txt') -Destination $package
Copy-Item -LiteralPath (Join-Path $root 'readme.md') -Destination $package
$revision=git -C $root rev-parse HEAD
@{version='2.3.0';source_commit=$revision;platform='Windows x64';cuda=$cudaVersion;gpu_architecture=$architecture;
    built_utc=[DateTime]::UtcNow.ToString('o');runtime_dlls=@('cublas64_13.dll','cublasLt64_13.dll','curand64_10.dll')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'BUILDINFO.json') -Encoding utf8
$manifest=Get-ChildItem -LiteralPath $package -File -Recurse | ForEach-Object {
    @{path=$_.FullName.Substring($package.Length+1); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash; bytes=$_.Length}
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $package 'SHA256.json') -Encoding utf8
$archive="$package.zip"
Compress-Archive -LiteralPath $package -DestinationPath $archive
Write-Output $archive
