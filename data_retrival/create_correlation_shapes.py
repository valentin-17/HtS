"""
Correlation patterns — 4 × 7 grid of 28 point clouds.

Reproduces the classic statistical illustration of Pearson correlation
(rows 1–2) and nonlinear / zero-correlation shapes (rows 3–4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Global settings ───────────────────────────────────────────────────────────
N = 500          # points per shape — adjust here
SEED = 42
HERE = Path(__file__).parent
OUTPUT_SHAPES_DIR = HERE / "data" / "shapes"
OUTPUT_PNG = HERE / "data" / "correlation_shapes.png"

rng = np.random.default_rng(SEED)

# dark blue (low t) → pale ice-blue (high t)
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dark_pale_blue", ["#08306b", "#c6dbef"]
)


# ── Helper ────────────────────────────────────────────────────────────────────
def _scale(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) * 100.0 if hi > lo else np.full_like(arr, 50.0)


def _make(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    xs, ys = _scale(x), _scale(y)
    return pd.DataFrame({"x": xs, "y": ys, "t": xs + ys})


def _make_iso(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Both axes share one scale factor so aspect ratio is preserved.
    Longer axis → [0, 100]; shorter axis centred at 50."""
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    span = max(x.max() - x.min(), y.max() - y.min())
    scale = 100.0 / span if span > 0 else 1.0
    xs = (x - cx) * scale + 50.0
    ys = (y - cy) * scale + 50.0
    return pd.DataFrame({"x": xs, "y": ys, "t": xs + ys})


# ── Row 1: bivariate normal distributions ─────────────────────────────────────
def gen_bivariate_normal(r: float, n: int = N) -> pd.DataFrame:
    if abs(r) == 1.0:
        x = rng.normal(0.0, 1.0, n)
        return _make(x, float(np.sign(r)) * x)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, r], [r, 1.0]], n)
    return _make(xy[:, 0], xy[:, 1])


# ── Row 2: perfect lines (r ≈ ±1) at fixed visual angles ────────────────────
def gen_line(angle_deg: float, n: int = N) -> pd.DataFrame:
    """Line at given angle (degrees from horizontal).

    Uses isotropic scaling (same factor for x and y) so the visual angle is
    preserved. Both axes are mapped via the same scale factor, centred at 50.
    """
    th = np.radians(angle_deg)
    t = np.linspace(-1.0, 1.0, n) + rng.normal(0.0, 1e-9, n)
    xs = t * np.cos(th) * 50.0 + 50.0   # [-1,1] → [0,100] along x
    ys = t * np.sin(th) * 50.0 + 50.0   # [-1,1] → [0,100] along y
    return pd.DataFrame({"x": xs, "y": ys, "t": xs + ys})


# ── Rows 3 & 4: nonlinear shapes ──────────────────────────────────────────────
def gen_sine_wave(noise: float = 0.05, cycles: float = 2.0, n: int = N) -> pd.DataFrame:
    x = rng.uniform(-np.pi * cycles, np.pi * cycles, n)
    return _make(x, np.sin(x) + rng.normal(0.0, noise, n))


def gen_diffuse_cloud(spread: float = 1.2, n: int = N) -> pd.DataFrame:
    return _make(rng.normal(0.0, spread, n), rng.normal(0.0, spread, n))


def gen_ellipse(a: float = 1.4, b: float = 0.38, n: int = N) -> pd.DataFrame:
    """Uniformly filled elongated ellipse (football shape)."""
    xs, ys, k = [], [], 0
    while k < n:
        u = rng.uniform(-a, a, n * 2)
        v = rng.uniform(-b, b, n * 2)
        m = (u / a)**2 + (v / b)**2 <= 1.0
        xs.append(u[m]); ys.append(v[m]); k += int(m.sum())
    return _make(np.concatenate(xs)[:n], np.concatenate(ys)[:n])


def gen_rotated_square(theta_deg: float = 45.0, n: int = N) -> pd.DataFrame:
    """Uniform on [0,1]² rotated by theta_deg — fills most of [0,100]²."""
    u = rng.uniform(0.0, 1.0, n)
    v = rng.uniform(0.0, 1.0, n)
    th = np.radians(theta_deg)
    x = u * np.cos(th) - v * np.sin(th)
    y = u * np.sin(th) + v * np.cos(th)
    return _make(x, y)


def gen_stretched_gaussian(sigma_x: float = 2.0, sigma_y: float = 0.25,
                            theta_deg: float = 30.0, n: int = N) -> pd.DataFrame:
    """Dense stretched Gaussian (σx ≫ σy) rotated by theta_deg."""
    x = rng.normal(0.0, sigma_x, n)
    y = rng.normal(0.0, sigma_y, n)
    th = np.radians(theta_deg)
    xr = x * np.cos(th) - y * np.sin(th)
    yr = x * np.sin(th) + y * np.cos(th)
    return _make_iso(xr, yr)


def gen_diamond(noise: float = 0.08, n: int = N) -> pd.DataFrame:
    """Uniform distribution inside a diamond (|x| + |y| ≤ 1)."""
    xs, ys, k = [], [], 0
    while k < n:
        u = rng.uniform(-1.0, 1.0, n * 2)
        v = rng.uniform(-1.0, 1.0, n * 2)
        m = np.abs(u) + np.abs(v) <= 1.0
        xs.append(u[m]); ys.append(v[m]); k += int(m.sum())
    u = np.concatenate(xs)[:n]
    v = np.concatenate(ys)[:n]
    return _make(u + rng.normal(0.0, noise, n), v + rng.normal(0.0, noise, n))


def gen_lens(height: float = 0.22, noise: float = 0.01, n: int = N) -> pd.DataFrame:
    """Horizontal lens: |y| ≤ height × (1 − x²). Uses isotropic scaling."""
    xs, ys, k = [], [], 0
    while k < n:
        u = rng.uniform(-1.0, 1.0, n * 2)
        v = rng.uniform(-height, height, n * 2)
        m = np.abs(v) <= height * (1.0 - u**2)
        xs.append(u[m]); ys.append(v[m]); k += int(m.sum())
    u = np.concatenate(xs)[:n] + rng.normal(0.0, noise, n)
    v = np.concatenate(ys)[:n] + rng.normal(0.0, noise, n)
    return _make_iso(u, v)


def gen_parabola(noise: float = 0.06, n: int = N) -> pd.DataFrame:
    x = rng.uniform(-1.0, 1.0, n)
    return _make(x, x**2 + rng.normal(0.0, noise, n))


def gen_x_shape(noise: float = 0.05, n: int = N) -> pd.DataFrame:
    """y = x² (opens up) and y = −x² (opens down), crossing at center (50, 50)."""
    h = n // 2
    x1 = rng.uniform(-1.0, 1.0, h)
    x2 = rng.uniform(-1.0, 1.0, n - h)
    x = np.concatenate([x1, x2])
    y = np.concatenate([
        x1**2 + rng.normal(0.0, noise, h),
        -x2**2 + rng.normal(0.0, noise, n - h),
    ])
    return _make(x, y)


def gen_circle(noise: float = 0.07, n: int = N) -> pd.DataFrame:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = 1.0 + rng.normal(0.0, noise, n)
    return _make(r * np.cos(theta), r * np.sin(theta))


def gen_four_clusters(spread: float = 0.12, n: int = N) -> pd.DataFrame:
    per, rem = divmod(n, 4)
    xs, ys = [], []
    for i, (cx, cy) in enumerate([(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]):
        cnt = per + (1 if i < rem else 0)
        xs.append(rng.normal(cx, spread, cnt))
        ys.append(rng.normal(cy, spread, cnt))
    return _make(np.concatenate(xs), np.concatenate(ys))


# ── Build catalog: 4 rows × 7 columns = 28 shapes ────────────────────────────
SHAPES: list = (
    # Row 1 — bivariate normals, r = 1, 0.7, 0.3, 0.1, −0.3, −0.8, −1
    [gen_bivariate_normal(r) for r in [1.0, 0.7, 0.3, 0.1, -0.3, -0.8, -1.0]]
    # Row 2 — perfect lines, angles steep-positive → steep-negative (degrees)
    + [gen_line(a) for a in [75.0, 55.0, 30.0, 0.0, -30.0, -55.0, -75.0]]
    # Row 3 — noisy nonlinear shapes
    + [
        gen_sine_wave(noise=0.30, cycles=2.0),  # W / noisy sine
        gen_rotated_square(theta_deg=45.0),
        gen_diamond(noise=0.08),
        gen_parabola(noise=0.10),
        gen_x_shape(noise=0.08),
        gen_circle(noise=0.10),
        gen_four_clusters(spread=0.15),
    ]
    # Row 4 — clean nonlinear shapes (same types, less noise)
    + [
        gen_sine_wave(noise=0.04, cycles=2.0),  # clean sine wave
        gen_lens(height=0.45, noise=0.02),
        gen_lens(height=0.22, noise=0.01),
        gen_parabola(noise=0.02),
        gen_x_shape(noise=0.02),
        gen_circle(noise=0.02),
        gen_four_clusters(spread=0.05),
    ]
)

FORM_IDS = [f"r{row}_c{col}" for row in range(1, 5) for col in range(1, 8)]


# ── CSV export ─────────────────────────────────────────────────────────────────
OUTPUT_SHAPES_DIR.mkdir(parents=True, exist_ok=True)

for fid, shape in zip(FORM_IDS, SHAPES):
    path = OUTPUT_SHAPES_DIR / f"{fid}.csv"
    shape[["x", "y", "t"]].to_csv(path, index=False)

print(f"CSVs → {OUTPUT_SHAPES_DIR}/ ({len(SHAPES)} files)")


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 7, figsize=(14, 8), facecolor="white")

for ax, shape in zip(axes.flat, SHAPES):
    ax.scatter(
        shape["x"], shape["y"],
        c=shape["t"], cmap=CMAP,
        s=2.0, linewidths=0, alpha=0.8,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

plt.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"PNG → {OUTPUT_PNG}")
