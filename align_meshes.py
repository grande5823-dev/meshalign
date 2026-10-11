import sys
import argparse
from pathlib import Path
import meshalign as ma

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "mech_parts"
DEFAULT_OUT = ROOT / "results"

def main():
    parser = argparse.ArgumentParser(description="Align and measure meshes using meshalign.")
    parser.add_argument("--ref", type=Path, default=DEFAULT_DATA / "Connector.stl", help="Path to reference mesh")
    parser.add_argument("--test", type=Path, default=DEFAULT_DATA / "Connector_test.stl", help="Path to test mesh")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--no-icp", action="store_true", help="Disable ICP registration")

    args = parser.parse_args()

    print(f"ref : {args.ref}")
    print(f"test: {args.test}")
    print(f"out : {args.out}\n")

    # Ensure output directory exists
    args.out.mkdir(parents=True, exist_ok=True)

    # Full pipeline: register (auto local/global) + measure + write artifacts.
    mae_um, p95_um, char_um, rel_mae, rel_p95 = ma.run_pair_relative(
        str(args.ref), str(args.test), str(args.out), 
        use_icp=not args.no_icp, histogram=True,
        save_per_sample=False
    )

    print(
        f"\nDone. MAE = {mae_um:.1f} um (p95 {p95_um:.1f}), "
        f"relative DFT = {rel_mae:.3f}% of the {char_um/1000:.1f} mm "
        f"characteristic length. Artifacts in {args.out}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())