"""Documento mínimo con tres salidas: Markdown, HTML autocontenido y PDF (matplotlib)."""

from __future__ import annotations

import base64
import html
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]


@dataclass
class Block:
    kind: str
    text: str = ""
    items: List[str] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    path: Optional[Path] = None


class Document:
    def __init__(self, title: str, subtitle: str = "") -> None:
        self.title = title
        self.subtitle = subtitle
        self.blocks: List[Block] = []

    def h1(self, text: str) -> None:
        self.blocks.append(Block("h1", text))

    def h2(self, text: str) -> None:
        self.blocks.append(Block("h2", text))

    def p(self, text: str) -> None:
        self.blocks.append(Block("p", text))

    def bullets(self, items: Sequence[str]) -> None:
        self.blocks.append(Block("bullets", items=[str(i) for i in items]))

    def table(self, headers: Sequence[str], rows: Sequence[Sequence], caption: str = "") -> None:
        self.blocks.append(
            Block(
                "table",
                text=caption,
                headers=[str(h) for h in headers],
                rows=[[str(c) for c in r] for r in rows],
            )
        )

    def image(self, path: PathLike, caption: str = "") -> None:
        if Path(path).exists():
            self.blocks.append(Block("image", text=caption, path=Path(path)))

    def pagebreak(self) -> None:
        self.blocks.append(Block("pagebreak"))


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(doc: Document, path: PathLike) -> Path:
    """Escribe el Markdown; las imágenes se enlazan con rutas relativas a ``path``."""
    path = Path(path)
    out: List[str] = [f"# {doc.title}", ""]
    if doc.subtitle:
        out += [f"_{doc.subtitle}_", ""]
    for b in doc.blocks:
        if b.kind == "h1":
            out += [f"## {b.text}", ""]
        elif b.kind == "h2":
            out += [f"### {b.text}", ""]
        elif b.kind == "p":
            out += [b.text, ""]
        elif b.kind == "bullets":
            out += [f"- {i}" for i in b.items] + [""]
        elif b.kind == "table":
            if b.text:
                out += [f"**{b.text}**", ""]
            out.append("| " + " | ".join(_md_cell(h) for h in b.headers) + " |")
            out.append("|" + "|".join("---" for _ in b.headers) + "|")
            out += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in b.rows]
            out.append("")
        elif b.kind == "image" and b.path is not None:
            rel = os.path.relpath(b.path, path.parent).replace("\\", "/")
            out += [f"![{b.text}]({rel})", ""]
    path.write_text("\n".join(out), encoding="utf-8")
    return path


_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;
color:#1f2933;line-height:1.5}
h1{border-bottom:3px solid #2f6fdb;padding-bottom:.3rem}
h2{margin-top:2rem;border-bottom:1px solid #d5dbe3;padding-bottom:.2rem;color:#2f6fdb}
h3{margin-top:1.4rem}
table{border-collapse:collapse;margin:.8rem 0;font-size:.85rem}
th,td{border:1px solid #d5dbe3;padding:.25rem .6rem;text-align:left}
th{background:#eef3fb}tr:nth-child(even) td{background:#fafbfd}
figure{margin:1rem 0;text-align:center}img{max-width:100%}
figcaption{font-size:.8rem;color:#52606d}.sub{color:#52606d}
"""


def to_html(doc: Document, path: PathLike) -> Path:
    """HTML autocontenido: las imágenes van embebidas en base64."""
    esc = html.escape

    def inline(text: str) -> str:
        return re.sub(
            r"`([^`]+)`", r"<code>\1</code>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text))
        )

    out: List[str] = [
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>",
        f"<title>{esc(doc.title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{esc(doc.title)}</h1>",
    ]
    if doc.subtitle:
        out.append(f"<p class='sub'>{esc(doc.subtitle)}</p>")
    for b in doc.blocks:
        if b.kind == "h1":
            out.append(f"<h2>{esc(b.text)}</h2>")
        elif b.kind == "h2":
            out.append(f"<h3>{esc(b.text)}</h3>")
        elif b.kind == "p":
            out.append(f"<p>{inline(b.text)}</p>")
        elif b.kind == "bullets":
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in b.items) + "</ul>")
        elif b.kind == "table":
            cap = f"<caption><b>{esc(b.text)}</b></caption>" if b.text else ""
            head = "".join(f"<th>{esc(h)}</th>" for h in b.headers)
            body = "".join(
                "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in b.rows
            )
            out.append(f"<table>{cap}<tr>{head}</tr>{body}</table>")
        elif b.kind == "image" and b.path is not None:
            data = base64.b64encode(b.path.read_bytes()).decode("ascii")
            out.append(
                f"<figure><img alt='{esc(b.text)}' src='data:image/png;base64,{data}'>"
                f"<figcaption>{esc(b.text)}</figcaption></figure>"
            )
    out.append("</body></html>")
    path = Path(path)
    path.write_text("".join(out), encoding="utf-8")
    return path


_PAGE_W, _PAGE_H, _MARGIN = 8.27, 11.69, 0.7
_BLUE = "#2f6fdb"


def _is_number(text: str) -> bool:
    try:
        float(text.replace("%", "").replace(",", ""))
    except ValueError:
        return False
    return True


class _PdfWriter:
    """Maquetación en flujo sobre páginas A4 con matplotlib (sin dependencias extra)."""

    def __init__(self, pdf, title: str) -> None:
        self.pdf = pdf
        self.title = title
        self.page = 0
        self.fig = None
        self.y = 0.0
        self.new_page()

    @property
    def width(self) -> float:
        return _PAGE_W - 2 * _MARGIN

    def new_page(self) -> None:
        import matplotlib.pyplot as plt

        self.close_page()
        self.page += 1
        self.fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
        self.y = _PAGE_H - _MARGIN

    def close_page(self) -> None:
        import matplotlib.pyplot as plt

        if self.fig is None:
            return
        self.fig.text(
            0.5,
            0.4 / _PAGE_H,
            f"{self.title}  -  {self.page}",
            ha="center",
            fontsize=7,
            color="#7b8794",
        )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def room(self, height: float) -> None:
        if self.y - height < _MARGIN:
            self.new_page()

    def text(self, text, size=9.0, weight="normal", color="#1f2933", x=0.0) -> None:
        char_w = size * 0.53 / 72
        cols = max(10, int((self.width - x) / char_w))
        line_h = size * 1.4 / 72
        for para in str(text).split("\n"):
            for line in textwrap.wrap(para, cols) or [""]:
                self.room(line_h)
                self.y -= line_h
                self.fig.text(
                    (_MARGIN + x) / _PAGE_W,
                    self.y / _PAGE_H,
                    line,
                    fontsize=size,
                    fontweight=weight,
                    color=color,
                    va="bottom",
                )

    def gap(self, inches: float) -> None:
        self.y -= inches

    def rule(self, color=_BLUE, lw=1.0) -> None:
        from matplotlib.lines import Line2D

        self.fig.add_artist(
            Line2D(
                [_MARGIN / _PAGE_W, (_PAGE_W - _MARGIN) / _PAGE_W],
                [self.y / _PAGE_H] * 2,
                color=color,
                lw=lw,
            )
        )

    def table(self, headers: List[str], rows: List[List[str]], caption: str) -> None:
        size, row_h = 6.8, 0.165
        char_w = size * 0.6 / 72
        total_chars = int(self.width / char_w) - len(headers)
        widths = [
            max([len(headers[i])] + [len(r[i]) for r in rows if i < len(r)])
            for i in range(len(headers))
        ]
        while sum(widths) > total_chars:
            widths[widths.index(max(widths))] -= 1
        if caption:
            self.gap(0.05)
            self.room(row_h * 3)
            self.text(caption, size=8.5, weight="bold")

        def draw(cells: List[str], bold: bool) -> None:
            self.room(row_h)
            self.y -= row_h
            x = _MARGIN
            for cell, w in zip(cells, widths):
                shown = cell if len(cell) <= w else cell[: max(0, w - 1)] + "…"
                right = not bold and _is_number(cell)
                self.fig.text(
                    (x + (w * char_w if right else 0)) / _PAGE_W,
                    self.y / _PAGE_H,
                    shown,
                    fontsize=size,
                    family="DejaVu Sans Mono",
                    fontweight="bold" if bold else "normal",
                    ha="right" if right else "left",
                    va="bottom",
                )
                x += (w + 1) * char_w

        draw(headers, True)
        self.rule("#9aa5b1", 0.5)
        for r in rows:
            if self.y - row_h < _MARGIN:
                self.new_page()
                draw(headers, True)
                self.rule("#9aa5b1", 0.5)
            draw(r + [""] * (len(headers) - len(r)), False)
        self.gap(0.12)

    def image(self, path: Path, caption: str) -> None:
        import matplotlib.pyplot as plt

        img = plt.imread(str(path))
        h_px, w_px = img.shape[0], img.shape[1]
        w = min(self.width, 6.0 if w_px >= h_px else 4.4)
        h = w * h_px / w_px
        h_max = _PAGE_H - 2 * _MARGIN - 0.6
        if h > h_max:
            w, h = w * h_max / h, h_max
        self.room(h + 0.35)
        self.y -= h
        left = (_MARGIN + (self.width - w) / 2) / _PAGE_W
        ax = self.fig.add_axes([left, self.y / _PAGE_H, w / _PAGE_W, h / _PAGE_H])
        ax.imshow(img)
        ax.axis("off")
        if caption:
            self.gap(0.05)
            self.text(caption, size=7.5, color="#52606d")
        self.gap(0.1)


def _plain(text: str) -> str:
    """Quita el énfasis Markdown (``**negrita**`` y ``código``) para el PDF."""
    return text.replace("**", "").replace("`", "")


def to_pdf(doc: Document, path: PathLike) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    with PdfPages(str(path)) as pdf:
        w = _PdfWriter(pdf, doc.title)
        w.gap(0.1)
        w.text(doc.title, size=20, weight="bold", color=_BLUE)
        if doc.subtitle:
            w.text(doc.subtitle, size=9, color="#52606d")
        w.gap(0.05)
        w.rule()
        w.gap(0.12)
        for b in doc.blocks:
            if b.kind == "h1":
                w.gap(0.18)
                w.room(0.6)
                w.text(b.text, size=13, weight="bold", color=_BLUE)
                w.rule("#d5dbe3", 0.6)
                w.gap(0.06)
            elif b.kind == "h2":
                w.gap(0.08)
                w.room(0.5)
                w.text(b.text, size=10.5, weight="bold")
            elif b.kind == "p":
                w.text(_plain(b.text))
                w.gap(0.06)
            elif b.kind == "bullets":
                for item in b.items:
                    w.text("•  " + _plain(item), x=0.1)
                w.gap(0.06)
            elif b.kind == "table":
                w.table(b.headers, b.rows, b.text)
            elif b.kind == "image" and b.path is not None:
                w.image(b.path, b.text)
            elif b.kind == "pagebreak":
                w.new_page()
        w.close_page()
    return path


def write_all(doc: Document, directory: PathLike, stem: str = "report") -> Dict[str, Path]:
    """Escribe ``<stem>.md``, ``<stem>.html`` y ``<stem>.pdf`` en ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "markdown": to_markdown(doc, directory / f"{stem}.md"),
        "html": to_html(doc, directory / f"{stem}.html"),
        "pdf": to_pdf(doc, directory / f"{stem}.pdf"),
    }
