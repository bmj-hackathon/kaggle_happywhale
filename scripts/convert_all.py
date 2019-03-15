from pathlib import Path

import jupytext

references_path = Path.cwd() / "reference notebooks"
assert references_path.exists()
references_scripts_path = Path.cwd() / "reference scripts"
assert references_scripts_path.exists()

notebooks = [p for p in references_path.iterdir() if p.suffix == ".ipynb"]
scripts = [p for p in references_path.iterdir() if p.suffix == ".py"]

for f in notebooks:

    converted_files = [p.stem for p in references_scripts_path.iterdir()]
    if f.stem in converted_files:
        print("already exists, skip", f.stem)
        continue

    print(f.stem)
    nb = jupytext.readf(f)
# jupytext.readf(nb_file, fmt=None)