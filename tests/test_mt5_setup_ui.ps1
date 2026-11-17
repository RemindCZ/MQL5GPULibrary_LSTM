param(
    [Parameter(Mandatory=$true)][string]$Setup,
    [Parameter(Mandatory=$true)][string]$Scratch
)
# Exercise the installer form in its actual .NET Framework / STA runtime.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$asm = [Reflection.Assembly]::LoadFrom($Setup)
$type = $asm.GetType('Remind.DNForecast.Setup.InstallerWindow', $true)
$flags = [Reflection.BindingFlags]'Instance,NonPublic,Public'
$folders = [Collections.Generic.List[string]]::new()
foreach ($name in @('Broker One','Broker Two')) {
    $dir = [IO.Path]::Combine($Scratch,$name)
    [IO.Directory]::CreateDirectory([IO.Path]::Combine($dir,'MQL5')) | Out-Null
    $folders.Add($dir)
}
$log = [IO.Path]::Combine($Scratch,'interactive.log')
function Window([string]$target) {
    $argsForWindow = [object[]]@($folders.PSObject.BaseObject,$null,$log.PSObject.BaseObject,$false)
    if ($target) { $argsForWindow[1] = $target.PSObject.BaseObject }
    $form = [Activator]::CreateInstance($type,$flags,$null,$argsForWindow,$null)
    $form.StartPosition = 'Manual'
    $form.Location = [Drawing.Point]::new(-10000,-10000)
    $form.ShowInTaskbar = $false
    $form.Show()
    [Windows.Forms.Application]::DoEvents()
    return $form
}
function Control($form,$name) { $form.Controls.Find($name,$true)[0] }
function Assert($value,$message) { if (!$value) { throw $message } }
$firstPath = [IO.Path]::Combine($folders[0],'MQL5','Libraries','DNForecast.dll')
$secondPath = [IO.Path]::Combine($folders[1],'MQL5','Libraries','DNForecast.dll')
$form = Window
try {
    $choice = Control $form 'Terminals'
    $install = Control $form 'Install'
    Assert ($choice.Items.Count -eq 2 -and $choice.SelectedIndex -eq -1 -and !$install.Enabled) 'Multiple targets must require a selection.'
    $choice.SelectedIndex = 0
    Assert ((Control $form 'TargetPath').Text -eq $folders[0]) 'Incorrect first target.'
    $choice.SelectedIndex = 1
    Assert ((Control $form 'TargetPath').Text -eq $folders[1] -and $install.Enabled) 'Incorrect second target.'
    (Control $form 'Cancel').PerformClick()
    Assert (!(Test-Path $firstPath) -and !(Test-Path $secondPath) -and !(Test-Path $log)) 'Selection or cancellation wrote to disk.'
} finally { $form.Dispose() }

$form = Window ($folders[1] + '\MQL5')
try {
    Assert ((Control $form 'TargetPath').Text -eq $folders[1]) 'Explicit MQL5 directory should resolve to its data root.'
    Assert (!(Test-Path $log)) 'An explicit GUI target bypassed confirmation.'
    (Control $form 'Install').PerformClick()
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    while (!(Control $form 'Install').Enabled -and [DateTime]::UtcNow -lt $deadline) {
        [Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 35
    }
    Assert ((Control $form 'Install').Enabled -and (Test-Path $secondPath) -and !(Test-Path $firstPath)) 'Installation must affect only the selected profile.'
    Assert ($type.GetField('Result',$flags).GetValue($form) -eq 0) 'Installer reported a failure.'
    (Control $form 'Install').PerformClick()
} finally { $form.Dispose() }

$folders.Clear()
$form = Window
try {
    Assert (!(Control $form 'Install').Enabled -and (Control $form 'Browse').Enabled) 'Manual browse should be available without any detected broker.'
} finally { $form.Close(); $form.Dispose() }

Write-Output 'PASS: explicit GUI confirmation, switching terminals, cancellation, selected profile only and manual folder option.'
