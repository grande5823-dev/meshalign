"""meshalign command-line interface.

    meshalign register REF TEST [-o OUT.stl] [--global|--local] [--no-write]
    meshalign measure  REF TEST --out DIR [--no-histogram] [--no-per-sample]
    meshalign run      REF TEST --out DIR        # register + measure (run_pair)
    meshalign chamfer  A B                       # symmetric chamfer distance
    meshalign ensemble M1 M2 [more...]           # no-reference consistency
    meshalign batch    --refs DIR --tests DIR --out DIR

The parsers are generated from ``meshalign.commands.COMMANDS`` (the same
registry that drives the webapp API and form), and every subcommand executes
through ``meshalign.commands.execute``.
"""
import argparse
import sys

from .commands import COMMANDS, execute


def _add_param(sp, p):
    name, t = p["name"], p["type"]
    dest = name
    flag = "--" + name.replace("_", "-")
    kw = dict(help=p["help"], dest=dest)
    if p.get("positional"):
        if t == "datasets":
            sp.add_argument(name, nargs="+", help=p["help"])
        else:
            sp.add_argument(name, help=p["help"])
        return
    names = ([p["short"]] if p.get("short") else []) + [flag]
    if t == "bool":
        if p["default"] is False:
            sp.add_argument(*names, action="store_true", default=False, **kw)
        else:   # None (preset decides) or True -> --x / --no-x, default preserved
            sp.add_argument(*names, action=argparse.BooleanOptionalAction,
                            default=p["default"], **kw)
    elif t == "int":
        sp.add_argument(*names, type=int, default=p["default"],
                        required=bool(p.get("required")), **kw)
    elif t == "float":
        sp.add_argument(*names, type=float, default=p["default"], **kw)
    elif t == "enum":
        sp.add_argument(*names, choices=p["choices"], default=p["default"], **kw)
    else:   # str | path | group
        sp.add_argument(*names, default=p["default"],
                        required=bool(p.get("required")), **kw)
    for alias, value in (p.get("flags") or {}).items():
        sp.add_argument(alias, dest=dest, action="store_const", const=value,
                        help=f"shorthand for {flag} {value}")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="meshalign",
        description="Rigid mesh/STL alignment and surface-error evaluation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, c in COMMANDS.items():
        sp = sub.add_parser(name, help=c["summary"])
        for p in c["params"]:
            _add_param(sp, p)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    params = {k: v for k, v in vars(args).items() if k != "cmd"}
    execute(args.cmd, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
