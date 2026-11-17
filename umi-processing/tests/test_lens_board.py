"""umiproc lens-board: the ChArUco board geometry and its printable PDF (page size, physical size, detectability)."""

import re
import shutil
import subprocess
import zlib

import cv2
import numpy as np
import pytest

from umiproc.cli import main
from umiproc.lens import board

HAVE_POPPLER = shutil.which("pdfinfo") is not None and shutil.which("pdftoppm") is not None
needs_poppler = pytest.mark.skipif(not HAVE_POPPLER, reason="pdfinfo / pdftoppm (poppler-utils) not installed")


def test_board_parameters():
    b = board.make_board()
    assert b.getChessboardSize() == (8, 6)
    assert b.getSquareLength() == pytest.approx(0.030) and b.getMarkerLength() == pytest.approx(0.022)
    assert not b.getLegacyPattern()
    assert list(b.getIds().ravel()) == list(range(24))
    assert b.getDictionary().markerSize == 5                 # 5x5, never the 4x4 dictionary of the finger tags
    info = board.board_info()
    assert info["dictionary"] == "DICT_5X5_100" and info["inner_corners"] == 35


@pytest.mark.parametrize("scale", [1, 2, 7])
def test_vector_geometry_is_opencvs_board(scale):
    """black_rects_mm (what the PDF draws) rasterised = CharucoBoard.generateImage, pixel for pixel."""
    p = board.EXACT_PX_PER_MM * scale
    w, h = int(round(240 * p)), int(round(180 * p))
    ref = board.make_board().generateImage((w, h), marginSize=0, borderBits=1)
    assert np.array_equal(board.board_image(scale), ref)


def test_finger_tag_dictionary_does_not_see_the_board():
    img = board.board_image(2, 14.0)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(img)
    assert ids is None or len(ids) == 0


def _content(pdf: bytes) -> str:
    m = re.search(rb"/FlateDecode >>\nstream\n(.*?)\nendstream", pdf, re.S)
    return zlib.decompress(m.group(1)).decode("ascii")


@pytest.mark.parametrize("paper", ["letter", "a4"])
def test_pdf_structure_and_text(paper):
    pdf = board.board_pdf(paper)
    assert pdf.startswith(b"%PDF-1.6") and pdf.rstrip().endswith(b"%%EOF")
    assert board.board_pdf(paper) == pdf                         # deterministic
    xref = int(re.search(rb"startxref\n(\d+)", pdf).group(1))
    assert pdf[xref:xref + 4] == b"xref"
    for k, off in enumerate(re.findall(rb"(\d{10}) 00000 n", pdf), start=1):
        assert pdf[int(off):].startswith(f"{k} 0 obj".encode())   # every xref offset points at its object
    assert b"/PrintScaling /None" in pdf
    text = _content(pdf)
    assert "Print at 100 % / actual size" in text and "Check: one square = 30.0 mm" in text
    assert "DICT_5X5_100" in text and "8 x 6 squares, square 30.0 mm, marker 22.0 mm" in text
    assert "100 mm scale bar" in text
    assert re.search(r" 100 2 re S", text)                       # the 100 mm bar outline (in mm units)


def _pdfinfo_size(path):
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out)
    return float(m.group(1)), float(m.group(2)), out


@needs_poppler
@pytest.mark.parametrize("paper,size_pt", [("letter", (792.0, 612.0)), ("a4", (841.89, 595.276))])
def test_pdf_page_size(tmp_path, paper, size_pt, capsys):
    out = tmp_path / f"b_{paper}.pdf"
    assert main(["lens-board", "--out", str(out), "--paper", paper]) == 0
    assert "print it at 100 %" in capsys.readouterr().out
    w, h, info = _pdfinfo_size(out)
    assert (w, h) == pytest.approx(size_pt, abs=0.01)
    assert re.search(r"Pages:\s+1\n", info)


@needs_poppler
@pytest.mark.parametrize("paper", ["letter", "a4"])
def test_rendered_pdf_is_detected_at_true_size(tmp_path, paper):
    """pdftoppm at 150 dpi -> OpenCV finds all 35 corners, one square apart = 30 mm at 150 dpi."""
    pdf = board.write_board_pdf(tmp_path / "b.pdf", paper)
    dpi = 150
    subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", str(pdf), str(tmp_path / "r")], check=True)
    img = cv2.imread(str(next(tmp_path.glob("r*.pgm"))), cv2.IMREAD_GRAYSCALE)
    pw, ph = board.PAPERS[paper]
    assert img.shape == pytest.approx((ph / 25.4 * dpi, pw / 25.4 * dpi), abs=1.5)
    cc, ci, _, mi = cv2.aruco.CharucoDetector(board.make_board()).detectBoard(img)
    assert len(ci) == 35 and len(mi) == 24
    c = dict(zip(ci.ravel(), cc.reshape(-1, 2).astype(np.float64)))
    steps = [np.linalg.norm(c[i + 1] - c[i]) for i in range(35) if i % 7 < 6]
    steps += [np.linalg.norm(c[i + 7] - c[i]) for i in range(28)]
    square_px = 30.0 / 25.4 * dpi
    assert np.median(steps) == pytest.approx(square_px, rel=0.005)
    assert max(abs(s - square_px) for s in steps) < 1.0
    # the board is centred horizontally: its outer corners 1 square beyond the outer inner corners
    left = c[0][0] - square_px
    right = c[6][0] + square_px
    assert left == pytest.approx(img.shape[1] - right, abs=2.0)


def test_default_output_name(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["lens-board"]) == 0
    assert (tmp_path / "umiproc_lens_board.pdf").read_bytes() == board.board_pdf("letter")
    assert "letter landscape" in capsys.readouterr().out


def test_unwritable_output(tmp_path, capsys):
    (tmp_path / "file").write_text("x")
    assert main(["lens-board", "--out", str(tmp_path / "file" / "b.pdf")]) == 1
    assert "cannot write" in capsys.readouterr().err
