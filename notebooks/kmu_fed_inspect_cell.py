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

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    flat_files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith(IMAGE_EXTS))

    if class_parent is None and flat_files:
        # No class-folder structure at all - class/subject must be encoded in the filename
        # itself instead (e.g. "01_AN_mr_001.jpg"). Parse it empirically rather than guessing
        # what each underscore-separated field means.
        print(f"No class folders found, but {len(flat_files)} image files sit directly in "
              f"{ROOT} - the class (and likely subject) must be encoded in the filename.\n")

        from collections import Counter, defaultdict

        parsed = [(f, Path(f).stem.split("_")) for f in flat_files]
        field_counts = Counter(len(p) for _, p in parsed)
        print("=" * 70)
        print("FILENAME STRUCTURE")
        print("=" * 70)
        print(f"Underscore-separated field-count distribution: {dict(field_counts)}")
        modal_n = field_counts.most_common(1)[0][0]
        odd_ones = [f for f, p in parsed if len(p) != modal_n]
        print(f"Modal pattern: {modal_n} fields. Files NOT matching it ({len(odd_ones)}): {odd_ones[:20]}\n")

        matching = [(f, p) for f, p in parsed if len(p) == modal_n]
        per_field_values = defaultdict(Counter)
        for _, p in matching:
            for i, val in enumerate(p):
                per_field_values[i][val] += 1
        for i in range(modal_n):
            vals = per_field_values[i]
            preview = dict(sorted(vals.items())[:15])
            print(f"Field {i}: {len(vals)} unique value(s) - {preview}"
                  f"{' ...' if len(vals) > 15 else ''}")

        # Cross-tab of field 0 (likely subject) x field 1 (likely class), the KMU-FED norm
        print("\n" + "=" * 70)
        print("FIELD 0 x FIELD 1 CROSS-TAB (rows = field 0, columns = field 1)")
        print("=" * 70)
        cross = defaultdict(Counter)
        for _, p in matching:
            if len(p) >= 2:
                cross[p[0]][p[1]] += 1
        rows, cols = sorted(cross), sorted({c for r in cross.values() for c in r})
        print("".ljust(10) + "".join(c.ljust(8) for c in cols))
        for r in rows:
            print(r.ljust(10) + "".join(str(cross[r].get(c, 0)).ljust(8) for c in cols))
        print(f"\n{len(rows)} unique field-0 values (candidate subject IDs): {rows}")
        print(f"{len(cols)} unique field-1 values (candidate class codes): {cols}")

        # Field 0 (numeric) vs. field 2, when field 2 also looks identity-like (few unique
        # values, all files partitioned cleanly) - resolves whether field 0 and field 2 are
        # two redundant aliases for the same subject, or genuinely independent variables.
        if modal_n >= 3:
            print("\n" + "=" * 70)
            print("FIELD 2 x FIELD 0 (does each field-2 value map to exactly one field-0 value?)")
            print("=" * 70)
            f2_to_f0 = defaultdict(Counter)
            for _, p in matching:
                f2_to_f0[p[2]][p[0]] += 1
            for f2 in sorted(f2_to_f0):
                mapping = dict(f2_to_f0[f2])
                tag = "CLEAN (1 field-0 value)" if len(mapping) == 1 else "MIXED (multiple field-0 values!)"
                print(f"  field2={f2!r}: field0 counts = {mapping}  -> {tag}")

        print("\n" + "=" * 70)
        print("IMAGE PROPERTIES (one sample per field-1 value)")
        print("=" * 70)
        seen, samples = set(), []
        for f, p in matching:
            cls = p[1] if len(p) >= 2 else "?"
            if cls in seen:
                continue
            seen.add(cls)
            img = Image.open(ROOT / f)
            arr = np.array(img)
            print(f"  {cls} ({f}): mode={img.mode}, size={img.size}, dtype={arr.dtype}, "
                  f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
            samples.append((cls, img.convert("RGB")))

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
            print(f"\nSaved a one-sample-per-value contact sheet to {out_path}")
            print("Download it from the notebook's Output pane: does it look like true IR/"
                  "night-vision (grayscale, glowing eyes, flat lighting) or an ordinary daylight photo?")
    elif class_parent is None:
        print("Could not auto-detect a class-folder level, and no image files sit directly in "
              "ROOT either - inspect the tree above by hand.")
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
