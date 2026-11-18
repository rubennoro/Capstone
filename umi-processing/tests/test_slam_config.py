"""umiproc.slam.config: loading configs/slam/rig_pi_v3.yaml, validation errors that name the field, mask_keep."""

import copy

import numpy as np
import pytest
import yaml

from umiproc.dp.slam.mask import keep_mask
from umiproc.paths import repo_config
from umiproc.slam.config import SlamConfigError, load_slam_config, mask_keep, parse_slam_config

REPO_FILE = repo_config("slam", "rig_pi_v3.yaml")


@pytest.fixture(scope="module")
def doc():
    return yaml.safe_load(REPO_FILE.read_text())


def test_repo_file_loads():
    cfg = load_slam_config(REPO_FILE)
    assert cfg.schema == "umiproc.slam.v1" and cfg.rig == "rig_pi_v3"
    assert cfg.mask.resolution == (1280, 960)
    assert len(cfg.mask.polygons) == 2 and all(len(p) >= 3 for p in cfg.mask.polygons)
    assert all(isinstance(v, float) for p in cfg.mask.polygons for pt in p for v in pt)
    assert cfg.mask.polygons[0][0] == (176.0, 960.0)
    assert cfg.mask.source and "traced" in cfg.mask.source
    assert (cfg.table_tag.dictionary, cfg.table_tag.id, cfg.table_tag.size_m) == ("DICT_4X4_50", 13, 0.16)
    assert (cfg.orb.n_features, cfg.orb.scale_factor, cfg.orb.n_levels, cfg.orb.ini_th_fast,
            cfg.orb.min_th_fast) == (1500, 1.2, 8, 20, 7)
    with pytest.raises(Exception):
        cfg.rig = "other"                                    # frozen


def _bad(doc, edit, match):
    d = copy.deepcopy(doc)
    edit(d)
    with pytest.raises(SlamConfigError, match=match):
        parse_slam_config(d, "x.yaml")


def _set(path, value):
    def edit(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return edit


@pytest.mark.parametrize("path,value,match", [
    (("schema",), "umiproc.slam.v2", r"x\.yaml: schema: expected 'umiproc\.slam\.v1'"),
    (("rig",), "bad id!", r"rig: invalid rig id"),
    (("mask", "resolution"), [1280], r"mask\.resolution: expected \[width, height\]"),
    (("mask", "resolution"), [1280, True], r"mask\.resolution"),
    (("mask", "polygons"), [], r"mask\.polygons: expected a non-empty list"),
    (("mask", "polygons"), [[[0, 0], [10, 0]]], r"mask\.polygons\[0\]: expected a polygon: a list of at least 3"),
    (("mask", "polygons"), [[[0, 0], [10, 0], [5]]], r"mask\.polygons\[0\]\[2\]: expected a point"),
    (("mask", "polygons"), [[[0, 0], [10, 0], [5, "a"]]], r"mask\.polygons\[0\]\[2\]\[1\]: expected a number"),
    (("mask", "polygons"), [[[0, 0], [1281, 0], [5, 5]]], r"mask\.polygons\[0\]\[1\]: point \[1281, 0\] is outside"),
    (("mask", "polygons"), [[[0, 0], [10, 0], [20, 0]]], r"mask\.polygons\[0\]: degenerate polygon"),
    (("mask", "source"), 3, r"mask\.source: expected a string"),
    (("table_tag", "dictionary"), "DICT_9X9_1", r"table_tag\.dictionary: not an OpenCV predefined"),
    (("table_tag", "dictionary"), "detectMarkers", r"table_tag\.dictionary: not an OpenCV predefined"),
    (("table_tag", "id"), 50, r"table_tag\.id: DICT_4X4_50 has ids 0\.\.49, got 50"),
    (("table_tag", "id"), -1, r"table_tag\.id: must be >= 0"),
    (("table_tag", "id"), True, r"table_tag\.id: expected an integer"),
    (("table_tag", "size_m"), 0, r"table_tag\.size_m: must be > 0"),
    (("table_tag", "size_m"), float("nan"), r"table_tag\.size_m: must be finite"),
    (("orb", "n_features"), 0, r"orb\.n_features: must be > 0"),
    (("orb", "n_features"), 1500.0, r"orb\.n_features: expected an integer"),
    (("orb", "n_levels"), -2, r"orb\.n_levels: must be > 0"),
    (("orb", "scale_factor"), 0, r"orb\.scale_factor: must be > 0"),
    (("orb", "scale_factor"), 1.0, r"orb\.scale_factor: must be > 1"),
    (("orb", "min_th_fast"), 25, r"orb\.min_th_fast: must be <= ini_th_fast"),
])
def test_invalid_fields_are_named(doc, path, value, match):
    _bad(doc, _set(path, value), match)


@pytest.mark.parametrize("section", [None, "mask", "table_tag", "orb"])
def test_unknown_keys_are_errors(doc, section):
    def edit(d):
        (d if section is None else d[section])["extra"] = 1
    _bad(doc, edit, (f"{section}." if section else "") + r"extra: unknown key")


@pytest.mark.parametrize("key", ["schema", "rig", "mask", "table_tag", "orb"])
def test_missing_sections(doc, key):
    _bad(doc, lambda d: d.pop(key), rf"{key}: missing")


def test_file_errors(tmp_path, doc):
    with pytest.raises(SlamConfigError, match="cannot read"):
        load_slam_config(tmp_path / "rig_pi_v3.yaml")
    (tmp_path / "rig_pi_v3.yaml").write_text("mask: [unclosed")
    with pytest.raises(SlamConfigError, match="invalid YAML"):
        load_slam_config(tmp_path / "rig_pi_v3.yaml")
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(SlamConfigError, match="does not match the file name 'other'"):
        load_slam_config(tmp_path / "other.yaml")
    (tmp_path / "rig_pi_v3.slam.yaml").write_text(yaml.safe_dump(doc))      # a copy next to the rig file
    assert load_slam_config(tmp_path / "rig_pi_v3.slam.yaml").rig == "rig_pi_v3"


def test_ints_and_floats_are_the_same_mask(doc):
    d = copy.deepcopy(doc)
    d["mask"]["polygons"] = [[[float(x), float(y)] for x, y in p] for p in d["mask"]["polygons"]]
    assert parse_slam_config(d).mask == parse_slam_config(doc).mask


def test_mask_keep():
    cfg = load_slam_config(REPO_FILE)
    k = mask_keep(cfg)
    assert k.shape == (960, 1280) and k.dtype == np.uint8 and set(np.unique(k)) == {0, 255}
    assert np.array_equal(k, keep_mask(cfg.mask.polygons, (1280, 960), (1280, 960)))
    assert k[0, 0] == 255 and k[100, 640] == 255                   # the top of the image is kept
    assert k[959, 671] == 0 and k[600, 640] == 0                   # the gripper (bottom centre) is masked
    assert k[959, 0] == 255                                        # bottom-left corner: outside the polygons
    assert 0.25 < float(np.mean(k == 0)) < 0.35
    small = mask_keep(cfg, (640, 480))
    assert small.shape == (480, 640)
    assert abs(float(np.mean(small == 0)) - float(np.mean(k == 0))) < 0.01
    with pytest.raises(ValueError):
        k[0, 0] = 1                                                # cached and read-only
