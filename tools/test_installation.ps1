param([Parameter(Mandatory=$true)][string]$PackageRoot)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$package=[IO.Path]::GetFullPath($PackageRoot)
$target=Join-Path $root ('tmp\MT5 test '+[guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path (Join-Path $target 'MQL5') -Force | Out-Null
$installer=Join-Path $package 'tools\Install-MT5.ps1'
& $installer -TerminalDataPath $target -PackageRoot $package
$files=@(Get-ChildItem -LiteralPath (Join-Path $package 'MQL5') -Recurse -File)
foreach($file in $files) {
    $relative=$file.FullName.Substring($package.Length+1)
    if((Get-FileHash -LiteralPath $file.FullName).Hash -ne (Get-FileHash -LiteralPath (Join-Path $target $relative)).Hash) {
        throw "Installed hash mismatch: $relative"
    }
}
$refused=$false
try { & $installer -TerminalDataPath $target -PackageRoot $package }
catch {
    if($_.Exception.Message -notlike 'Already exists:*') { throw }
    $refused=$true
}
if(-not $refused) { throw 'Installer must refuse an implicit overwrite' }
& $installer -TerminalDataPath $target -PackageRoot $package -Force
Write-Output "PASS: $($files.Count) installed files match, implicit overwrite refused, explicit upgrade accepted."
Write-Output "Isolated test data: $target"
