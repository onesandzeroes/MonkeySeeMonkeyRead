import os
import subprocess

figs = os.listdir('figures')

for fname in figs:
    base, ext = os.path.splitext(fname)
    if ext == '.svg':
        subprocess.call(
            ['convert', '-size', '500x500', 'figures/' + fname, 'figures/' + base + '.png']
        )
