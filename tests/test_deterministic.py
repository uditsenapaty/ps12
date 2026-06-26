"""Deterministic battery — the green CPU gate before any GPU spend.

Every test runs on REAL code paths (no mocks): normalization round-trip, tiling reconstruct-identity,
metric sanity, NetCDF I/O round-trip, filename timestamp parsing, and a real classical optical-flow
interpolation that must beat naive blending on a known translation. Seeded; sub-second; CPU-only.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data import normalize as norm
from src.data import tiling
from src.data.triplets import build_triplets, index_frames, parse_timestamp
from src.eval import metrics

RNG = np.random.default_rng(12345)


# ---- normalization --------------------------------------------------------------------------------
def test_bt_norm_roundtrip_exact():
    bt = RNG.uniform(180.0, 330.0, size=(64, 80)).astype(np.float32)
    back = norm.norm_to_bt(norm.bt_to_norm(bt))
    assert np.allclose(bt, back, atol=1e-2)


def test_bt_norm_clips_out_of_range():
    bt = np.array([[100.0, 400.0]], dtype=np.float32)  # below/above physical range
    x = norm.bt_to_norm(bt)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_bt_norm_nan_preserved():
    bt = np.full((8, 8), 250.0, np.float32)
    bt[0, 0] = np.nan
    x = norm.bt_to_norm(bt)
    assert np.isnan(x[0, 0])
    filled, mask = norm.fill_invalid(x)
    assert filled[0, 0] == 0.0 and mask[0, 0] == False  # noqa: E712


# ---- tiling ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape,tile,overlap", [((200, 200), 64, 16), ((257, 301), 128, 32), ((100, 64), 64, 8)])
def test_tiling_reconstruct_identity(shape, tile, overlap):
    img = RNG.random(shape).astype(np.float32)
    tiles, pos, sh = tiling.tile_image(img, tile, overlap)
    recon = tiling.untile_image(tiles, pos, sh, overlap)
    assert recon.shape == shape
    assert np.allclose(img, recon, atol=1e-5)


def test_tiling_covers_edges():
    img = RNG.random((150, 150)).astype(np.float32)
    tiles, pos, sh = tiling.tile_image(img, 64, 16)
    recon = tiling.untile_image(tiles, pos, sh, 16)
    assert np.allclose(img[-1, -1], recon[-1, -1], atol=1e-5)


# ---- metric sanity --------------------------------------------------------------------------------
def test_metric_identity():
    x = RNG.random((48, 48)).astype(np.float32)
    assert metrics.mse(x, x) == 0.0
    assert np.isinf(metrics.psnr(x, x))
    assert abs(metrics.ssim(x, x) - 1.0) < 1e-6
    assert metrics.mae_kelvin(x * 150 + 180, x * 150 + 180) == 0.0


def test_metric_masks_nan():
    a = RNG.random((16, 16)).astype(np.float32)
    b = a.copy()
    a[0, 0] = np.nan  # invalid in one -> excluded
    assert metrics.mse(a, b) == 0.0


# ---- filename timestamp parsing -------------------------------------------------------------------
def test_parse_goes_timestamp():
    name = "OR_ABI-L1b-RadF-M6C13_G19_s20261750010204_e20261750019512_c20261750019578.nc"
    ts = parse_timestamp(name, "goes19")
    assert (ts.year, ts.month, ts.hour, ts.minute) == (2026, 6, 0, 10)  # DOY 175 of 2026 = Jun 24


def test_parse_insat_timestamp():
    ts = parse_timestamp("3RIMG_24JUN2026_0014_L1C_SGP_V01R00.h5", "insat3dr")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute) == (2026, 6, 24, 0, 14)


def test_build_triplets_spacing():
    from datetime import datetime, timedelta
    base = datetime(2026, 6, 24, 0, 0)
    indexed = [(base + timedelta(minutes=10 * i), f"f{i}.nc") for i in range(5)]
    trips = build_triplets(indexed, step_minutes=10)
    assert len(trips) == 3  # (0,1,2),(1,2,3),(2,3,4)


def test_build_anytime_configurable_grid():
    import pytest
    from datetime import datetime, timedelta
    from src.data.triplets import build_anytime_samples
    base = datetime(2026, 6, 24, 0, 0)
    indexed = [(base + timedelta(minutes=10 * i), f"f{i}.nc") for i in range(13)]  # 0..120 min, 10-min cadence

    # dt = 0.5  -> ONLY the midpoint, across gap sizes (20/40/60 min)
    mid = build_anytime_samples(indexed, cadence_minutes=10, time_step=0.5, gap_levels=3)
    assert mid and {t for *_, t in mid} == {0.5}

    # dt = 0.25 -> t in {0.25, 0.5, 0.75}, spans 40/80/120 (base = 4*cadence)
    q = build_anytime_samples(indexed, cadence_minutes=10, time_step=0.25, gap_levels=3)
    assert {t for *_, t in q} == {0.25, 0.5, 0.75}
    for p0, p2, gt, t in q:
        assert p0 != p2 and 0.0 < t < 1.0
    # off-grid frames are skipped: with dt=0.5 a 40-min span uses +20 (t=0.5), never +10
    s40 = [s for s in mid if s[0] == "f0.nc" and s[1] == "f4.nc"]   # 0->40 min span
    assert s40 and all(gt == "f2.nc" for _, _, gt, _ in s40)        # only frame@20, not frame@10

    # time_step must evenly divide 1
    with pytest.raises(ValueError):
        build_anytime_samples(indexed, cadence_minutes=10, time_step=0.3)


def test_unet_vfi_is_t_conditioned():
    """t must actually change the output (it's an input feature, not just a post-hoc flow scale)."""
    torch = pytest.importorskip("torch")
    from src.models.unet_vfi import build_net
    net = build_net()(base=8).eval()
    i0, i1 = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        a = net(i0, i1, 0.2)
        b = net(i0, i1, 0.8)
        assert a.shape == (1, 1, 64, 64)
        assert float((a - b).abs().mean()) > 1e-4
        # per-sample tensor t (arbitrary-time batch) returns one prediction per item
        pred = net(i0.repeat(3, 1, 1, 1), i1.repeat(3, 1, 1, 1), torch.tensor([0.25, 0.5, 0.75]))
        assert pred.shape == (3, 1, 64, 64)


# ---- real classical optical-flow interpolation ----------------------------------------------------
def _texture(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w]
    return (0.5 + 0.5 * np.sin(xx / 7.0) * np.cos(yy / 9.0)).astype(np.float32)


def test_classical_static_scene_is_identity():
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from src.models.classical import ClassicalFlowInterpolator
    img = _texture()
    out = ClassicalFlowInterpolator().interpolate(img, img, 0.5)
    assert np.mean(np.abs(out - img)) < 1e-3


def test_classical_beats_blend_on_translation():
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from src.models.classical import ClassicalFlowInterpolator, LinearBlendInterpolator
    base = _texture()
    shift = 6
    f0 = base
    f1 = np.roll(base, shift, axis=1)
    true_mid = np.roll(base, shift // 2, axis=1)
    sl = slice(16, -16)  # interior, avoid wrap boundary
    flow_pred = ClassicalFlowInterpolator().interpolate(f0, f1, 0.5)
    blend = LinearBlendInterpolator().interpolate(f0, f1, 0.5)
    e_flow = metrics.mse(flow_pred[sl, sl], true_mid[sl, sl])
    e_blend = metrics.mse(blend[sl, sl], true_mid[sl, sl])
    assert e_flow < e_blend  # motion compensation beats naive averaging


# ---- NetCDF round-trip + end-to-end interpolate ---------------------------------------------------
def test_nc_write_read_roundtrip(tmp_path):
    pytest.importorskip("xarray")
    from datetime import datetime
    from src.data.ncio import read_nc, write_nc
    bt = RNG.uniform(200, 300, (32, 40)).astype(np.float32)
    bt[0, :2] = np.nan
    p = write_nc(tmp_path / "f.nc", bt, datetime(2026, 6, 24, 0, 15), source="goes19", band="C13")
    bt2, t2, attrs = read_nc(p)
    assert bt2.shape == bt.shape
    assert np.allclose(np.nan_to_num(bt), np.nan_to_num(bt2), atol=1e-3)
    assert attrs["band"] == "C13"


def test_interpolate_pair_bt_endtoend():
    pytest.importorskip("cv2")
    from src.infer.interpolate import interpolate_pair_bt
    from src.models.classical import ClassicalFlowInterpolator
    bt0 = _texture(96, 96) * 100 + 200
    bt1 = np.roll(bt0, 4, axis=1)
    bt0[0, 0] = np.nan  # off-disk pixel
    bt1[0, 0] = np.nan
    pred = interpolate_pair_bt(bt0, bt1, ClassicalFlowInterpolator(), 0.5, tile=64, overlap=16)
    assert pred.shape == bt0.shape
    assert np.isnan(pred[0, 0])                 # invalid preserved
    assert np.isfinite(pred[50, 50])            # valid filled
    assert 150 <= np.nanmin(pred) and np.nanmax(pred) <= 360
