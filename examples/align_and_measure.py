"""Worked end-to-end example: align + measure the bundled Connector bracket.

Run from the repo root:

    python examples/align_and_measure.py

Writes results into examples/out/.
"""
import sys
from pathlib import Path

import meshalign as ma

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mech_parts"
OUT = ROOT / "examples" / "out"


def main():
    ref = DATA / "Connector.stl"
    test = DATA / "Connector_test.stl"

    print(f"ref : {ref}")
    print(f"test: {test}")
    print(f"out : {OUT}\n")

    # Full pipeline: register (auto local/global) + measure + write artifacts.
    mae_um, p95_um, char_um, rel_mae, rel_p95 = ma.run_pair_relative(
        str(ref), str(test), str(OUT), use_icp=True, histogram=True,
        save_per_sample=False)

    print(
        f"\nDone. MAE = {mae_um:.1f} um (p95 {p95_um:.1f}), "
        f"relative DFT = {rel_mae:.3f}% of the {char_um/1000:.1f} mm "
        f"characteristic length. Artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
