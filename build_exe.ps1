param(
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = "python"
$AppName = "SD_BIN_Analysis"
$Entry = "gui.py"

Write-Host "Checking Python packages..."
& $Python -m pip install --upgrade pip
& $Python -m pip install pyinstaller numpy scipy matplotlib pandas openpyxl h5py hdf5plugin mne scikit-learn

$PythonExe = (Get-Command $Python).Source
$PythonRoot = Split-Path -Parent $PythonExe
$TclRoot = Join-Path $PythonRoot "tcl"
$TclSourceDir = Join-Path $TclRoot "tcl8.6"
$TkSourceDir = Join-Path $TclRoot "tk8.6"

foreach ($RequiredPath in @($TclSourceDir, $TkSourceDir)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Cannot find Tcl/Tk runtime data: $RequiredPath"
    }
}

$PyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", $AppName,
    "--collect-all", "scipy",
    "--collect-all", "matplotlib",
    "--collect-all", "mne",
    "--collect-all", "sklearn",
    "--hidden-import", "openpyxl",
    "--hidden-import", "snr_gui",
    "--hidden-import", "compare_bin_storage",
    "--hidden-import", "h5py",
    "--collect-all", "hdf5plugin",
    "--hidden-import", "pandas",
    "--hidden-import", "tkinter",
    "--hidden-import", "tkinter.ttk",
    "--hidden-import", "tkinter.filedialog",
    "--hidden-import", "tkinter.messagebox",
    "--hidden-import", "matplotlib.figure",
    "--hidden-import", "matplotlib.backends.backend_tkagg",
    "--hidden-import", "matplotlib.backends._backend_tk",
    "--hidden-import", "scipy.signal",
    "--hidden-import", "scipy.fft",
    "--hidden-import", "scipy.io",
    "--hidden-import", "scipy.interpolate",
    "--hidden-import", "scipy.stats"
)

if ($OneFile) {
    $PyInstallerArgs += "--onefile"
}

$PyInstallerArgs += @("--add-data", "$TclSourceDir;tcl_data")
$PyInstallerArgs += @("--add-data", "$TkSourceDir;tk_data")
$PyInstallerArgs += @("--add-data", "$TclSourceDir;_tcl_data")
$PyInstallerArgs += @("--add-data", "$TkSourceDir;_tk_data")
$PyInstallerArgs += @("--add-data", "$Root\gui_core.pyc;.")

$PyInstallerArgs += $Entry

Write-Host "Building $AppName..."
& $Python @PyInstallerArgs

if ($OneFile) {
    Write-Host ""
    Write-Host "Done: $Root\dist\$AppName.exe"
} else {
    $DistDir = Join-Path $Root "dist\$AppName"
    $InternalDir = Join-Path $DistDir "_internal"
    $TclDataDir = Join-Path $InternalDir "tcl_data"
    $TkDataDir = Join-Path $InternalDir "tk_data"
    $LegacyTclDataDir = Join-Path $InternalDir "_tcl_data"
    $LegacyTkDataDir = Join-Path $InternalDir "_tk_data"
    $ExePath = Join-Path $DistDir "$AppName.exe"

    foreach ($RequiredPath in @($ExePath, $TclDataDir, $TkDataDir, $LegacyTclDataDir, $LegacyTkDataDir)) {
        if (-not (Test-Path -LiteralPath $RequiredPath)) {
            throw "Build output is incomplete: missing $RequiredPath"
        }
    }

    $ZipPath = Join-Path $Root "dist\$AppName.zip"
    if (Test-Path -LiteralPath $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }
    Compress-Archive -Path $DistDir -DestinationPath $ZipPath -Force

    Write-Host ""
    Write-Host "Done: $Root\dist\$AppName\$AppName.exe"
    Write-Host "Copy the whole folder '$Root\dist\$AppName' to another computer."
    Write-Host "Or send '$ZipPath' and unzip it before running the EXE."
}

Write-Host ""
Write-Host "Note: Python, the BIN parser, and the HDF5 filter plugin are bundled. MATLAB is not required."
