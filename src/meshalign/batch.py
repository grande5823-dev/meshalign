"""Multiprocess batch runner: run_pair over many parts, near-linear speedup.

Determinism contract: each worker runs single-threaded (OMP_NUM_THREADS=1,
set in the pool initializer BEFORE open3d is imported in the spawned child)
and each pair is seeded (DEFAULT_SEED + index) -- so every pair's result is
bit-exact reproducible regardless of --jobs, and identical to a serial
OMP_NUM_THREADS=1 run. That is the metrology-grade property: batch results
do not depend on machine load or worker count.

CLI:

    meshalign batch --refs REF_DIR --tests TEST_DIR --out OUT_DIR [--jobs N]
                    [--mode default] [--relative] [--pattern '*.stl']

matches files by basename across the two directories (sorted order).
Python API: ``run_batch(pairs, out_dir, jobs=8, use_icp=True, ...)`` where
pairs is a list of (ref_path, test_path).
"""
import os
import sys


def _batch_init():
    """Pool initializer: pin the worker to one thread BEFORE any heavy import.

    With the 'spawn' start method the child starts clean; this runs before
    the first task (whose function lazily imports meshalign), so the OMP
    setting is effective for numpy/open3d in the worker.
    """
    os.environ["OMP_NUM_THREADS"] = "1"


def _batch_worker(task):
    """One (ref, test, out_dir, seed, use_icp, mode, relative) task -> dict.

    Lazy imports: this module must stay import-light so spawned children
    set OMP_NUM_THREADS before numpy/open3d load.
    """
    ref, test, out_dir, seed, use_icp, mode, relative = task
    import numpy as np
    from meshalign.pipeline import run_pair_relative, run_pair
    try:
        fn = run_pair_relative if relative else run_pair
        out = fn(ref, test, out_dir, use_icp=use_icp, seed=seed, mode=mode)
        mae_um, p95_um = out[0], out[1]      # run_pair_relative returns 5, run_pair 2
        return dict(ref=ref, test=test, mae_um=round(mae_um, 4),
                    p95_um=round(p95_um, 4), ok=True)
    except Exception as e:
        return dict(ref=ref, test=test, error=f"{type(e).__name__}: {e}", ok=False)


def _match_pairs(ref_dir, test_dir, pattern):
    """Filename-matched pairs across two directories (sorted by basename)."""
    import glob
    refs = {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(ref_dir, pattern)))}
    tests = {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(test_dir, pattern)))}
    common = sorted(set(refs) & set(tests))
    missing = (set(refs) ^ set(tests))
    if missing:
        print(f"[meshalign.batch] unmatched files (skipped): {sorted(missing)[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    return [(refs[b], tests[b]) for b in common]


def run_batch(pairs, out_dir, jobs=None, use_icp=True, mode="default",
              relative=True, seed_base=None, on_result=None):
    """Run ``run_pair``/``run_pair_relative`` over many pairs in parallel.

    ``pairs``: list of (ref_path, test_path). ``jobs`` None = min(32, cpus).
    Per-pair seed = DEFAULT_SEED + pair index (deterministic across job
    counts). Each worker is single-threaded; total throughput scales with
    ``jobs`` while every pair's result stays bit-exact reproducible
    (OMP_NUM_THREADS=1 + fixed seed).

    Returns a list of per-pair result dicts (same order as ``pairs``):
    ``{ref, test, mae_um, p95_um, ok}`` or ``{ref, test, error, ok: False}``.
    Also writes ``out_dir/batch_summary.csv``. ``on_result(i, n, result)`` (if
    given) is called in pair order as each result becomes available.
    """
    from meshalign.config import DEFAULT_SEED
    import csv
    from concurrent.futures import ProcessPoolExecutor

    base = DEFAULT_SEED if seed_base is None else int(seed_base)
    tasks = [(r, t, out_dir, base + i, use_icp, mode, relative)
             for i, (r, t) in enumerate(pairs)]
    os.makedirs(out_dir, exist_ok=True)

    if jobs is None:
        jobs = min(32, os.cpu_count() or 1)
    # ALL runs go through spawned workers (even jobs=1): the spawned child
    # imports open3d with OMP_NUM_THREADS=1 already set, so serial and
    # parallel runs are bit-identical AND independent of the parent's
    # threading state. An in-process path would inherit whatever threading
    # the parent already imported -- measured to break the determinism
    # contract (near-tied arbitrations wobble under multithreaded ICP).
    from multiprocessing import get_context
    _saved_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(max_workers=max(1, jobs),
                                 mp_context=get_context("spawn"),
                                 initializer=_batch_init) as ex:
            futures = [ex.submit(_batch_worker, t) for t in tasks]
            results = []
            for i, fut in enumerate(futures):
                r = fut.result()
                results.append(r)
                if on_result is not None:
                    on_result(i, len(tasks), r)
    finally:
        if _saved_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = _saved_omp

    ok = [r for r in results if r.get("ok")]
    csv_path = os.path.join(out_dir, "batch_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ref", "test", "mae_um", "p95_um", "ok", "error"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"[meshalign.batch] {len(ok)}/{len(results)} pairs OK; "
          f"summary: {csv_path}")
    return results


def main(argv=None):
    import argparse
    from meshalign.config import DEFAULT_SEED
    ap = argparse.ArgumentParser(prog="meshalign batch",
                                  description="Parallel align+measure over matched ref/test dirs.")
    ap.add_argument("--refs", required=True, help="reference directory")
    ap.add_argument("--tests", required=True, help="test directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--jobs", type=int, default=None,
                     help="worker processes (default min(32, cpus))")
    ap.add_argument("--mode", default="default", choices=["fast", "default", "accuracy"])
    ap.add_argument("--no-icp", action="store_true", help="measure as-is (no registration)")
    ap.add_argument("--absolute", action="store_true",
                    help="plain run_pair instead of size-normalized relative")
    ap.add_argument("--pattern", default="*", help="filename glob within both dirs")
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    pairs = _match_pairs(args.refs, args.tests, args.pattern)
    if not pairs:
        ap.error(f"no matched pairs between {args.refs} and {args.tests} "
                 f"(pattern {args.pattern!r})")
    print(f"[meshalign.batch] {len(pairs)} pairs, jobs="
          f"{args.jobs or min(32, __import__('os').cpu_count() or 1)}")
    run_batch(pairs, args.out, jobs=args.jobs, use_icp=not args.no_icp,
              mode=args.mode, relative=not args.absolute,
              seed_base=args.seed_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
