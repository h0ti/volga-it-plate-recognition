"""Convert the Markdown explanatory note to DOCX (python-docx).

Usage::

    python docs/build_docx.py docs/ЗАПИСКА.md docs/ЗАПИСКА.docx

Only the subset of Markdown used in the note is supported: headings, paragraphs,
bullet/numbered lists, pipe tables, fenced code blocks and **bold** runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm


def add_runs(paragraph, text: str) -> None:
    """Append text to a paragraph, honouring **bold** spans."""
    for index, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        run = paragraph.add_run(part)
        run.bold = index % 2 == 1


def flush_table(document, rows: list[str]) -> None:
    """Render collected pipe-table rows as a DOCX table."""
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    cells = [row for row in cells if not all(set(cell) <= {"-", ":", " "} for cell in row)]
    if not cells:
        return
    table = document.add_table(rows=len(cells), cols=len(cells[0]))
    table.style = "Table Grid"
    for row_index, row in enumerate(cells):
        for column_index, value in enumerate(row[: len(cells[0])]):
            paragraph = table.cell(row_index, column_index).paragraphs[0]
            add_runs(paragraph, value)
            if row_index == 0:
                for run in paragraph.runs:
                    run.bold = True


def convert(source: Path, destination: Path, author: str = "h0ti") -> None:
    """Convert the note and save it next to the Markdown source."""
    lines = source.read_text(encoding="utf-8").splitlines()
    document = Document()
    properties = document.core_properties
    properties.author = author
    properties.last_modified_by = author
    properties.comments = ""
    properties.title = "Пояснительная записка — распознавание ГРЗ РФ"
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for section in document.sections:
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(1.5)
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)

    table_rows: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_code() -> None:
        if not code_lines:
            return
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(6)
        run = paragraph.add_run("\n".join(code_lines))
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        code_lines.clear()

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("```"):
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(stripped)
            continue
        if stripped.startswith("|"):
            table_rows.append(stripped)
            continue
        if table_rows:
            flush_table(document, table_rows)
            table_rows.clear()
        if not stripped or stripped == "---":
            continue
        if stripped.startswith("# "):
            document.add_heading(stripped[2:], level=0)
        elif stripped.startswith("## "):
            document.add_heading(stripped[3:], level=1)
        elif stripped.startswith("### "):
            document.add_heading(stripped[4:], level=2)
        elif stripped.startswith("- "):
            add_runs(document.add_paragraph(style="List Bullet"), stripped[2:])
        elif re.match(r"^\d+\.\s", stripped):
            add_runs(document.add_paragraph(style="List Number"), re.sub(r"^\d+\.\s", "", stripped))
        else:
            add_runs(document.add_paragraph(), stripped)
    if table_rows:
        flush_table(document, table_rows)
    flush_code()

    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: build_docx.py <source.md> <destination.docx> [author]", file=sys.stderr)
        raise SystemExit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else "h0ti")
