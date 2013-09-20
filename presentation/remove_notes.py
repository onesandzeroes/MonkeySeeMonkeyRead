in_notes = False
with open('monkeysee.md') as full_file:
    with open('monkeysee-slides.md', 'w') as slide_file:
        for line in full_file:
            if line.strip() == '#### Notes':
                in_notes = not in_notes
                notes_line = True
            else:
                notes_line = False
            if not in_notes:
                if not notes_line:
                    slide_file.write(line)