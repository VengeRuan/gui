# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\ruanyuanjun\\AppData\\Local\\Programs\\Python\\Python311\\tcl\\tcl8.6', 'tcl_data'), ('C:\\Users\\ruanyuanjun\\AppData\\Local\\Programs\\Python\\Python311\\tcl\\tk8.6', 'tk_data'), ('C:\\Users\\ruanyuanjun\\AppData\\Local\\Programs\\Python\\Python311\\tcl\\tcl8.6', '_tcl_data'), ('C:\\Users\\ruanyuanjun\\AppData\\Local\\Programs\\Python\\Python311\\tcl\\tk8.6', '_tk_data'), ('D:\\E\\SD_copy_rawdataShow_SNR_GUI\\gui_core.pyc', '.')]
binaries = []
hiddenimports = ['openpyxl', 'snr_gui', 'compare_bin_storage', 'h5py', 'pandas', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'matplotlib.figure', 'matplotlib.backends.backend_tkagg', 'matplotlib.backends._backend_tk', 'scipy.signal', 'scipy.fft', 'scipy.io', 'scipy.interpolate', 'scipy.stats']
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mne')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('hdf5plugin')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SD_BIN_Analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SD_BIN_Analysis',
)
