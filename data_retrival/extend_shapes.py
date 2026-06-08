"""
For every original shape CSV (r{1-4}_c{1-7}.csv) in data/shapes/,
create one extended CSV per extra-column count (5, 10, 20, 50).
New columns are uniform random in [0, 200].

Usage:
    python data_retrival/extend_shapes.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

SHAPES_DIR   = Path(__file__).parent / "data" / "shapes"
EXTRA_COLS   = [5, 10, 20, 50]
VALUE_MIN    = 0
VALUE_MAX    = 200
SEED         = 42

rng = np.random.default_rng(SEED)

# collect only the original shape files (r{row}_c{col}.csv)
originals = sorted(
    p for p in SHAPES_DIR.glob("r[1-4]_c[1-7].csv")
    if p.stem.count("_") == 1   # exactly one underscore → original file
)

print(f"Found {len(originals)} original shape CSVs\n")

total = 0
for csv_path in originals:
    df = pd.read_csv(csv_path)
    n  = len(df)

    for n_extra in EXTRA_COLS:
        # generate random columns [0, 200]
        rand_data = rng.uniform(VALUE_MIN, VALUE_MAX, size=(n, n_extra))
        col_names = [f"f{i+1:02d}" for i in range(n_extra)]
        df_extra  = pd.concat(
            [df, pd.DataFrame(rand_data, columns=col_names)],
            axis=1,
        )

        out_name = f"{csv_path.stem}_ext{n_extra:02d}cols.csv"
        out_path = SHAPES_DIR / out_name
        df_extra.to_csv(out_path, index=False)
        total += 1

    print(f"  {csv_path.name}  ->  4 files  "
          f"({df.shape[1]} + 5/10/20/50 cols,  {n} rows)")

print(f"\nDone — {total} files written to {SHAPES_DIR}")
