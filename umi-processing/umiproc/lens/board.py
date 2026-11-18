"""The lens-calibration ChArUco board, and ``umiproc lens-board``: write it as a printable PDF.

The board: ``DICT_5X5_100`` markers (ids 0-23), 8 x 6 squares (landscape), square 30 mm, marker 22 mm, OpenCV's
current (non-legacy) layout, so ``cv2.aruco.CharucoBoard((8, 6), 0.030, 0.022, DICT_5X5_100)`` detects it
(:func:`make_board`). Not ``DICT_4X4_*``: the gripper's finger tags (ids 0/1) and UMI's table tag (id 13) are
``DICT_4X4_50`` and are in view while recording.

Board coordinates (OpenCV's ChArUco frame): origin at the board's top-left outer corner, x right, y down, z into the
board, metres in OpenCV, millimetres here. Black squares are those with ``(row + column)`` even; each marker sits in
a white square with a 4 mm white margin, its 7 x 7 cells (1-cell black border + 5 x 5 bits) from
``cv2.aruco.generateImageMarker``. :func:`black_rects_mm` is that geometry as rectangles; it is pixel-identical to
``CharucoBoard.generateImage`` (tests).

The PDF is written by hand (no PDF library): one page at the exact paper size (Letter or A4, landscape), the board
drawn as **vector** rectangles at its exact physical size (one filled path, so no seams between cells), the board
parameters, the print instruction and a 100 mm scale bar in Helvetica (a PDF base font, nothing embedded). The
catalog asks viewers to print without scaling (``/PrintScaling /None``), but the user still has to choose 100 % /
actual size and measure a square.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

__all__ = ["DICTIONARY", "SQUARES", "SQUARE_MM", "MARKER_MM", "MARKER_CELLS", "PAPERS", "DEFAULT_OUT",
           "aruco_dictionary", "make_board", "board_size_mm", "board_info", "black_rects_mm", "board_image",
           "board_pdf", "write_board_pdf", "add_arguments", "run"]

DICTIONARY = "DICT_5X5_100"
SQUARES = (8, 6)                  # (columns, rows): landscape
SQUARE_MM = 30.0
MARKER_MM = 22.0
MARKER_CELLS = 7                  # 5 x 5 bits + a 1-cell black border on each side
#: Landscape paper sizes (width, height) in mm.
PAPERS = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}
PAPER_NAMES = {"letter": "Letter", "a4": "A4"}
DEFAULT_OUT = "umiproc_lens_board.pdf"
PT_PER_MM = 72.0 / 25.4
#: Pixels per mm at which every board edge lands on a pixel boundary (cell 22/7 mm, margin 4 mm, square 30 mm).
EXACT_PX_PER_MM = 3.5


def aruco_dictionary():
    import cv2

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))


def make_board():
    """The board as OpenCV's ``CharucoBoard`` (lengths in metres)."""
    import cv2

    board = cv2.aruco.CharucoBoard(SQUARES, SQUARE_MM / 1000.0, MARKER_MM / 1000.0, aruco_dictionary())
    board.setLegacyPattern(False)
    return board


def board_size_mm() -> tuple[float, float]:
    return SQUARES[0] * SQUARE_MM, SQUARES[1] * SQUARE_MM


def board_info() -> dict:
    """The board parameters, as stored in reports and lens files."""
    n_markers = SQUARES[0] * SQUARES[1] // 2
    return {"type": "charuco", "dictionary": DICTIONARY, "squares": list(SQUARES), "square_m": SQUARE_MM / 1000.0,
            "marker_m": MARKER_MM / 1000.0, "marker_ids": [0, n_markers - 1], "legacy_pattern": False,
            "inner_corners": (SQUARES[0] - 1) * (SQUARES[1] - 1)}


def black_rects_mm() -> list[tuple[float, float, float, float]]:
    """Every black area of the board as ``(x, y, w, h)`` in board millimetres (x right, y down, origin top-left):
    the black squares, then each marker's black cells."""
    import cv2

    rects = [(c * SQUARE_MM, r * SQUARE_MM, SQUARE_MM, SQUARE_MM)
             for r in range(SQUARES[1]) for c in range(SQUARES[0]) if (r + c) % 2 == 0]
    board, d = make_board(), aruco_dictionary()
    cell = MARKER_MM / MARKER_CELLS
    for marker_id, corners in zip(board.getIds().ravel(), board.getObjPoints()):
        # The marker's top-left corner, snapped to its exact value (OpenCV stores float32 metres).
        x_cv, y_cv = float(corners[0][0]) * 1000.0, float(corners[0][1]) * 1000.0
        margin = (SQUARE_MM - MARKER_MM) / 2.0
        x0 = (x_cv // SQUARE_MM) * SQUARE_MM + margin
        y0 = (y_cv // SQUARE_MM) * SQUARE_MM + margin
        if abs(x0 - x_cv) > 1e-3 or abs(y0 - y_cv) > 1e-3:
            raise RuntimeError(f"unexpected ChArUco layout: marker {marker_id} at ({x_cv}, {y_cv}) mm")
        bits = cv2.aruco.generateImageMarker(d, int(marker_id), MARKER_CELLS, borderBits=1)
        for i in range(MARKER_CELLS):
            j = 0
            while j < MARKER_CELLS:                 # merge each row's black cells into runs
                if bits[i, j] != 0:
                    j += 1
                    continue
                k = j
                while k < MARKER_CELLS and bits[i, k] == 0:
                    k += 1
                rects.append((x0 + j * cell, y0 + i * cell, (k - j) * cell, cell))
                j = k
    return rects


def board_image(scale: int = 2, margin_mm: float = 0.0) -> np.ndarray:
    """The board rasterised at ``EXACT_PX_PER_MM * scale`` px/mm (uint8, 0 black / 255 white), with an optional
    white margin. Exact (no anti-aliasing needed) because every edge lands on a pixel boundary."""
    if int(scale) != scale or scale < 1:
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    p = EXACT_PX_PER_MM * int(scale)
    m = int(round(margin_mm * p))
    if abs(m - margin_mm * p) > 1e-9:
        raise ValueError(f"margin {margin_mm} mm is not a whole number of pixels at {p} px/mm")
    bw, bh = board_size_mm()
    img = np.full((int(round(bh * p)) + 2 * m, int(round(bw * p)) + 2 * m), 255, np.uint8)
    for x, y, w, h in black_rects_mm():
        x0, y0, x1, y1 = (int(round(v * p)) + m for v in (x, y, x + w, y + h))
        img[y0:y1, x0:x1] = 0
    return img


# --------------------------------------------------------------------------------------------- PDF

def _pdf_text(s: str) -> str:
    if any(ord(ch) > 126 or ord(ch) < 32 for ch in s):
        raise ValueError(f"PDF text must be printable ASCII: {s!r}")
    return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def _f(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _layout(paper: str) -> dict:
    """Positions in page millimetres, y up from the bottom edge (PDF's orientation)."""
    pw, ph = PAPERS[paper]
    bw, bh = board_size_mm()
    text_h, gap, bar_h = 8.0, 2.5, 7.0          # two text lines above, the scale bar below
    stack = text_h + gap + bh + gap + bar_h
    bottom = (ph - stack) / 2.0
    return {"page": (pw, ph), "board_x": (pw - bw) / 2.0, "board_top": bottom + bar_h + gap + bh,
            "board_bottom": bottom + bar_h + gap, "bar_y": bottom + 2.0, "text_y": bottom + stack - 3.0}


def _content(paper: str) -> bytes:
    L = _layout(paper)
    bx, top = L["board_x"], L["board_top"]
    bw, _ = board_size_mm()
    out = ["q", f"{_f(PT_PER_MM)} 0 0 {_f(PT_PER_MM)} 0 0 cm", "0 g"]
    # The board: every black area in ONE path, filled with the nonzero rule (re is always counter-clockwise), so
    # cells that touch render as one shape without hairline seams.
    for x, y, w, h in black_rects_mm():
        out.append(f"{_f(bx + x)} {_f(top - y - h)} {_f(w)} {_f(h)} re")
    out.append("f")
    # 100 mm scale bar: 10 mm segments, alternating black and white, outlined; tick marks at 0 and 100 mm.
    by, bar_t = L["bar_y"], 2.0
    for k in range(0, 10, 2):
        out.append(f"{_f(bx + 10.0 * k)} {_f(by)} 10 {_f(bar_t)} re")
    out.append("f")
    out += ["0.2 w", f"{_f(bx)} {_f(by)} 100 {_f(bar_t)} re S",
            f"{_f(bx)} {_f(by - 1.5)} m {_f(bx)} {_f(by + bar_t + 1.5)} l S",
            f"{_f(bx + 100.0)} {_f(by - 1.5)} m {_f(bx + 100.0)} {_f(by + bar_t + 1.5)} l S", "Q"]
    lo, hi = 0, SQUARES[0] * SQUARES[1] // 2 - 1
    lines = [
        ("F2", 10.0, bx, L["text_y"],
         f"umiproc lens calibration board: ChArUco {DICTIONARY} (ids {lo}-{hi}), {SQUARES[0]} x {SQUARES[1]} "
         f"squares, square {SQUARE_MM:.1f} mm, marker {MARKER_MM:.1f} mm"),
        ("F1", 9.0, bx, L["text_y"] - 4.2,
         f"Print at 100 % / actual size (not 'fit to page'). Check: one square = {SQUARE_MM:.1f} mm. "
         "Glue it flat onto something rigid."),
        ("F1", 9.0, bx + 104.0, L["bar_y"] + 0.3,
         f"<- 100 mm scale bar ({PAPER_NAMES[paper]} landscape). "
         "umiproc lens-board, DIRECTIONS.md section 2."),
    ]
    for font, size, x, y, text in lines:
        out.append(f"BT /{font} {_f(size)} Tf {_f(x * PT_PER_MM)} {_f(y * PT_PER_MM)} Td {_pdf_text(text)} Tj ET")
    return ("\n".join(out) + "\n").encode("ascii")


def board_pdf(paper: str = "letter") -> bytes:
    """The board PDF (one page, landscape ``paper``) as bytes. Deterministic: no dates in the file."""
    if paper not in PAPERS:
        raise ValueError(f"paper must be one of {sorted(PAPERS)}, got {paper!r}")
    pw, ph = (v * PT_PER_MM for v in PAPERS[paper])
    stream = zlib.compress(_content(paper), 9)
    title = _pdf_text(f"umiproc lens board {DICTIONARY} {SQUARES[0]}x{SQUARES[1]} {SQUARE_MM:g}/{MARKER_MM:g} mm "
                      f"({PAPER_NAMES[paper]} landscape)")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_f(pw)} {_f(ph)}] "
         f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>").encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode("ascii") + stream + b"\nendstream",
        f"<< /Title {title} /Producer (umiproc lens-board) >>".encode("ascii"),
    ]
    buf = bytearray(b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(buf))
        buf += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(buf)
    buf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii")
    for off in offsets:
        buf += f"{off:010d} 00000 n \n".encode("ascii")
    buf += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info {len(objs)} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("ascii")
    return bytes(buf)


def write_board_pdf(path: str | Path, paper: str = "letter") -> Path:
    """Write :func:`board_pdf` to ``path`` (parent directories are created). Returns the path."""
    path = Path(path)
    data = board_pdf(paper)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------------- command

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default=DEFAULT_OUT, metavar="FILE", help=f"PDF to write (default ./{DEFAULT_OUT})")
    p.add_argument("--paper", choices=sorted(PAPERS), default="letter", help="paper size, landscape "
                   "(default letter)")


def run(args) -> int:
    try:
        out = write_board_pdf(args.out, args.paper)
    except OSError as e:
        print(f"umiproc lens-board: cannot write {args.out}: {e}", file=sys.stderr)
        return 1
    bw, bh = board_size_mm()
    print(f"wrote {out} ({args.paper} landscape): ChArUco {DICTIONARY}, {SQUARES[0]} x {SQUARES[1]} squares of "
          f"{SQUARE_MM:.1f} mm (board {bw:.0f} x {bh:.0f} mm), markers {MARKER_MM:.1f} mm")
    print(f"print it at 100 % / actual size, then measure: one square must be {SQUARE_MM:.1f} mm and the scale bar "
          "100 mm (DIRECTIONS.md section 2.1)")
    return 0
