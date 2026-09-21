"""Build paper/paper.docx from paper/paper.md (the source of truth).

Handles only the Markdown subset used in paper.md: #/##/### headings, paragraphs with
**bold** / *italic* / `code`, bullet and numbered lists, pipe tables, images, and the
leading <!-- draft notes --> comment (shown as a grey italic note). Uses python-docx, which
is installed for the system Python (not the project venv):

    python3 paper/build_docx.py
"""
import re
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "paper.md", HERE / "paper.docx"
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`)")


def add_runs(paragraph, text: str, size: float = None, colour: RGBColor = None) -> None:
    """Add `text` to `paragraph`, turning **bold**, *italic* and `code` into formatted runs."""
    text = text.replace("&nbsp;", " ")
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2]); run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1]); run.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = paragraph.add_run(part[1:-1]); run.italic = True
        else:
            run = paragraph.add_run(part)
        if size:
            run.font.size = Pt(size)
        if colour is not None:
            run.font.color.rgb = colour


def add_table(doc: Document, rows: list) -> None:
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, cell_text in enumerate(header):
        cell = table.rows[0].cells[c]
        cell.text = ""
        add_runs(cell.paragraphs[0], f"**{cell_text}**", size=9)
    for r, row in enumerate(body, start=1):
        for c, cell_text in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.text = ""
            add_runs(cell.paragraphs[0], cell_text, size=9)
    doc.add_paragraph()


def main() -> None:
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(1)
        section.top_margin = section.bottom_margin = Inches(1)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    text = SRC.read_text(encoding="utf-8")
    grey = RGBColor(0x66, 0x66, 0x66)
    note = re.match(r"<!--(.*?)-->", text, re.S)
    if note:
        p = doc.add_paragraph()
        add_runs(p, "*DRAFT NOTES (delete before submission):* " + " ".join(note.group(1).split()), size=8.5, colour=grey)
        text = text[note.end():]

    lines, i = text.splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped or stripped == "---":
            i += 1
        elif stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            add_table(doc, [r for r in block if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)])
        elif re.match(r"^#{1,3} ", stripped):
            level = len(stripped.split()[0])
            heading = doc.add_heading(level=0 if level == 1 else level - 1)
            add_runs(heading, stripped[level + 1:])
            i += 1
        elif stripped.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            path = (HERE / m.group(2)).resolve()
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            with Image.open(path) as im:
                aspect = im.width / im.height
            # Sizing: wide figures (curves, charts, diagrams) use the full text width; near-square
            # ones (confusion matrices) are capped at 3.8" tall so a page doesn't end half empty;
            # the very tall Grad-CAM grid may use up to 7.5" of height so it stays on one page.
            max_height = 7.5 if aspect < 0.6 else 3.8
            width = min(6.5 if "architecture" in path.name else 6.0, max_height * aspect)
            p.add_run().add_picture(str(path), width=Inches(width))
            if m.group(1):
                cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(cap, f"*{m.group(1)}*", size=9.5)
            i += 1
        elif re.match(r"^(- |\d+\. )", stripped):
            style = "List Bullet" if stripped.startswith("- ") else "List Number"
            p = doc.add_paragraph(style=style)
            add_runs(p, re.sub(r"^(- |\d+\. )", "", stripped))
            i += 1
        else:
            para = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3} |\||!\[|- |\d+\. |---)", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_runs(p, " ".join(para))

    doc.save(OUT)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
