"""KMU-FED Step 0 inspection cell — paste this whole file into one Kaggle notebook cell and
run it, AFTER adding "anandpanajkar/kmu-fed" as a second "Add Input" (same way FER2013 was
added). This writes nothing outside /kaggle/working and does not need this repo cloned.

Paste the FULL printed output back, plus (if you can) the sample grid image it saves to
/kaggle/working/kmu_fed_sample_grid.png — download it from the notebook's Output pane.

This does not assume anything about KMU-FED's folder layout, class names, image mode, or
filename convention; it discovers and reports all of that, since dataset mirrors on Kaggle
sometimes differ from what papers describe.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_ROOT = Path("/kaggle/input")


def find_kmu_fed_root(input_root: Path) -> "Path | None":
    """Search under /kaggle/input for a folder that looks like it holds KMU-FED, without
    assuming a fixed nesting depth - some Kaggle environments mount datasets directly at
    /kaggle/input/<slug>/, others nest them under /kaggle/input/datasets/<owner>/<slug>/.
    """
    for dirpath, dirnames, _ in os.walk(input_root):
        depth = len(Path(dirpath).relative_to(input_root).parts)
        if depth > 5:
            dirnames[:] = []
            continue
        if "kmu" in Path(dirpath).name.lower() or "kmu-fed" in [d.lower() for d in dirnames]:
            for d in dirnames:
                if "kmu" in d.lower():
                    return Path(dirpath) / d
            if "kmu" in Path(dirpath).name.lower():
                return Path(dirpath)
    return None


ROOT = find_kmu_fed_root(INPUT_ROOT)
if ROOT is None:
    print("Could not find a 'kmu'-named folder anywhere under /kaggle/input (searched 5 levels deep).")
    print("Full tree under /kaggle/input (depth <= 5), so you can find it by eye:")
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
        dirnames.sort()
        depth = len(Path(dirpath).relative_to(INPUT_ROOT).parts)
        if depth > 5:
            dirnames[:] = []
            continue
        print(f"{'  ' * depth}{Path(dirpath).relative_to(INPUT_ROOT)}/  "
              f"({len(dirnames)} subdirs, {len(filenames)} files)")
    print("\n^ find the KMU-FED folder path above, then set ROOT = Path('/kaggle/input/<that path>') "
          "and re-run from the 'else' branch below by hand, or just paste the tree back to me.")
else:
    print(f"Using ROOT = {ROOT}\n")

    # ---- 1. Full directory tree (bounded depth, so this doesn't dump thousands of filenames)
    print("=" * 70)
    print("DIRECTORY TREE (depth <= 4)")
    print("=" * 70)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames.sort()
        rel = Path(dirpath).relative_to(ROOT)
        depth = len(rel.parts)
        if depth > 4:
            dirnames[:] = []
            continue
        print(f"{'  ' * depth}{rel}/  ({len(dirnames)} subdirs, {len(filenames)} files)")
        if filenames:
            print(f"{'  ' * depth}  e.g. {sorted(filenames)[:5]}")

    # ---- 2. Auto-detect the class-folder level: the shallowest directory whose immediate
    #         subdirectories mostly contain only image files (no further nesting).
    print("\n" + "=" * 70)
    print("AUTO-DETECTED CLASS FOLDERS")
    print("=" * 70)
    class_parent = None
    for dirpath, dirnames, filenames in sorted(os.walk(ROOT), key=lambda t: len(Path(t[0]).relative_to(ROOT).parts)):
        if len(dirnames) < 4:
            continue
        candidate_dirs = [Path(dirpath) / d for d in dirnames]
        leafy = [d for d in candidate_dirs if not any(os.path.isdir(d / f) for f in os.listdir(d)[:20])]
        if len(leafy) >= 4:
            class_parent = Path(dirpath)
            break

    if class_parent is None:
        print("Could not auto-detect a class-folder level - inspect the tree above by hand.")
    else:
        print(f"Class folders appear to live under: {class_parent}\n")
        class_dirs = sorted(d for d in os.listdir(class_parent) if (class_parent / d).is_dir())
        print(f"Class folder names ({len(class_dirs)} found): {class_dirs}\n")

        all_files_by_class = {}
        for cls in class_dirs:
            files = sorted(f for f in os.listdir(class_parent / cls) if not f.startswith("."))
            all_files_by_class[cls] = files
            print(f"  {cls}: {len(files)} files")
            print(f"    first 8 filenames: {files[:8]}")

        # ---- 3. Filename pattern check, for subject-identity encoding
        print("\n" + "=" * 70)
        print("FILENAME PATTERN CHECK (look for a repeating subject ID across classes)")
        print("=" * 70)
        for cls, files in all_files_by_class.items():
            print(f"  {cls}: {files[:12]}")

        # ---- 4. Image mode / pixel range / size, one sample per class
        print("\n" + "=" * 70)
        print("IMAGE PROPERTIES (one sample per class)")
        print("=" * 70)
        samples = []
        for cls, files in all_files_by_class.items():
            if not files:
                continue
            path = class_parent / cls / files[0]
            img = Image.open(path)
            arr = np.array(img)
            print(f"  {cls} ({files[0]}): mode={img.mode}, size={img.size}, dtype={arr.dtype}, "
                  f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
            samples.append((cls, img.convert("RGB")))

        # ---- 5. Save a contact sheet so you can eyeball whether these look like real IR/
        #         night-vision captures or ordinary daylight photos
        if samples:
            thumb = 150
            sheet = Image.new("RGB", (thumb * len(samples), thumb + 20), "white")
            from PIL import ImageDraw
            draw = ImageDraw.Draw(sheet)
            for i, (cls, img) in enumerate(samples):
                img = img.resize((thumb, thumb))
                sheet.paste(img, (i * thumb, 20))
                draw.text((i * thumb + 4, 2), cls[:18], fill="black")
            out_path = "/kaggle/working/kmu_fed_sample_grid.png"
            sheet.save(out_path)
            print(f"\nSaved a one-sample-per-class contact sheet to {out_path}")
            print("Download it from the notebook's Output pane and take a look: does it look like "
                  "true IR/night-vision (grayscale, glowing eyes, flat lighting) or an ordinary "
                  "daylight color photo?")

    # ---- 6. Total size on disk, for reference
    total_files = sum(len(f) for _, _, f in os.walk(ROOT))
    print(f"\nTotal files anywhere under ROOT: {total_files}")
