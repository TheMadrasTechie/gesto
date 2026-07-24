"""
Command line interface.

    gesto train static pose ./gesto_projects/postures
    gesto train sequence hands_one ./gesto_projects/signs --seq-len 30
    gesto detect static pose
    gesto detect sequence hands_one --source clip.mp4 --version 2
    gesto list
    gesto inspect ./gesto_projects/signs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, artifacts
from .regions import REGION_KEYS


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=artifacts.DEFAULT_ROOT,
                   help="artifact root folder (default: artifacts)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesto",
        description="Train and run gesture models from Gesto Labeller datasets.")
    parser.add_argument("--version", action="version",
                        version=f"gesto {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    t = sub.add_parser("train", help="train a model")
    t.add_argument("mode", choices=artifacts.MODES)
    t.add_argument("region", choices=REGION_KEYS)
    t.add_argument("project_dir", help="Gesto project folder (Copy path button)")
    t.add_argument("--seq-len", type=int, default=30,
                   help="frames per sequence (sequence mode, default 30)")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--val-split", type=float, default=0.2)
    t.add_argument("--small", dest="small", action="store_true", default=None,
                   help="force the lighter model")
    t.add_argument("--large", dest="small", action="store_false",
                   help="force the full-size model")
    t.add_argument("--raw", action="store_true",
                   help="data was captured with Gesto 'Normalise' unchecked")
    _add_common(t)

    # ---- detect ----
    d = sub.add_parser("detect", help="run live detection")
    d.add_argument("mode", choices=artifacts.MODES)
    d.add_argument("region", choices=REGION_KEYS)
    d.add_argument("--source", default="0", help="webcam index or video path")
    d.add_argument("--version", dest="run_version", default=None,
                   help="model version (default: newest)")
    d.add_argument("--threshold", type=float, default=0.5)
    d.add_argument("--smooth", type=int, default=5,
                   help="frames of agreement before committing a label")
    d.add_argument("--width", type=int, default=960, help="display width")
    d.add_argument("--no-mirror", dest="mirror", action="store_false",
                   default=None, help="don't mirror the webcam")
    _add_common(d)

    # ---- list ----
    ls = sub.add_parser("list", help="show trained models")
    _add_common(ls)

    # ---- inspect ----
    i = sub.add_parser("inspect", help="summarise a Gesto project's data")
    i.add_argument("project_dir")
    i.add_argument("--seq-len", type=int, default=30)

    return parser


def _cmd_train(args) -> int:
    from .train import train
    run = train(args.project_dir, args.region, args.mode, root=args.root,
                seq_len=args.seq_len, epochs=args.epochs,
                batch_size=args.batch_size, val_split=args.val_split,
                small=args.small, normalized=not args.raw)
    print(f"\nDetect with:  gesto detect {args.mode} {args.region}"
          f"{'' if args.root == artifacts.DEFAULT_ROOT else f' --root {args.root}'}")
    return 0


def _cmd_detect(args) -> int:
    from .detect import run as run_detect
    version = args.run_version
    if version is not None and str(version).isdigit():
        version = int(version)
    run_detect(args.mode, args.region, root=args.root, version=version,
               source=args.source, threshold=args.threshold,
               smooth=args.smooth, width=args.width, mirror=args.mirror)
    return 0


def _cmd_list(args) -> int:
    root = Path(args.root)
    if not root.exists():
        print(f"No artifacts yet under {root}/")
        return 0
    found = False
    for mode in artifacts.MODES:
        d = root / mode
        if not d.exists():
            continue
        runs = sorted(p for p in d.iterdir() if p.is_dir())
        if not runs:
            continue
        found = True
        print(f"{mode}/")
        for run in runs:
            try:
                meta = artifacts.load_meta(run)
                classes = ", ".join(meta.get("labels", []))
                extra = (f", seq_len={meta['seq_len']}"
                         if "seq_len" in meta else "")
                print(f"  {run.name:20} {meta.get('region','?'):10} "
                      f"{meta.get('samples','?')} samples{extra}  [{classes}]")
            except FileNotFoundError:
                print(f"  {run.name:20} (no labels.json)")
    if not found:
        print(f"No trained models under {root}/")
    return 0


def _cmd_inspect(args) -> int:
    import numpy as np
    from .data import project_meta

    project = Path(args.project_dir)
    meta = project_meta(project)
    if meta:
        print(f"Project: {meta.get('name', project.name)}  "
              f"region={meta.get('region')}  hands={meta.get('hands')}")
        print(f"Classes: {meta.get('classes', [])}")
    else:
        print(f"Project: {project.name}  (no project.json)")

    for mode in artifacts.MODES:
        root = project / "data" / mode
        if not root.exists():
            print(f"\n{mode}: none")
            continue
        labels = sorted(d.name for d in root.iterdir() if d.is_dir())
        print(f"\n{mode}:")
        dims = set()
        for name in labels:
            files = sorted((root / name).glob("*.npy"))
            if not files:
                print(f"  {name:16} 0")
                continue
            lengths = []
            for f in files:
                a = np.load(f)
                if a.ndim == 1:
                    a = a[None, :]
                lengths.append(a.shape[0])
                dims.add(a.shape[1])
            if mode == "sequence":
                usable = sum(1 for n in lengths if n >= args.seq_len)
                print(f"  {name:16} {len(files):4} clips, "
                      f"{usable} usable at seq_len={args.seq_len}, "
                      f"frames {min(lengths)}-{max(lengths)}")
            else:
                print(f"  {name:16} {len(files):4} samples")
        if dims:
            print(f"  feature dim: {sorted(dims)}"
                  + ("   MIXED — recapture consistently!" if len(dims) > 1 else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "train": _cmd_train,
        "detect": _cmd_detect,
        "list": _cmd_list,
        "inspect": _cmd_inspect,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
