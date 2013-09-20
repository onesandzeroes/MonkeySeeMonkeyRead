python remove_notes.py
pandoc --standalone --mathjax --slide-level=1 monkeysee-slides.md -t slidy -o monkeysee-slidy.html
pandoc --slide-level=1 monkeysee-slides.md -t beamer -o monkeysee-beamer.pdf
pandoc monkeysee.md -o monkeysee.pdf
