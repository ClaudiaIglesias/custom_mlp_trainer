# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from kivy_deps import sdl2, glew

a = Analysis(['mainprogram.py'],
             pathex=['C:\\Users\\claudia\\anaconda3\\share\\kivy-examples\\demo\\AI'],
             binaries=[],
             datas=[('redneuronallab.kv', '.'), ('images', 'images')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='mainprogram',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               a.datas,
                *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               upx_exclude=[],
               name='mainprogram')
