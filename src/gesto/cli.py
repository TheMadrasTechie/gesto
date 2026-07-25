"""
Command line interface.

Two styles, both available:

General form (mode and region as arguments):
    gesto train static pose ./gesto_projects/postures
    gesto detect sequence hands_one --source clip.mp4

Per-combination form (one command per mode+region):
    gesto train-static-pose ./gesto_projects/postures --epochs 250
    gesto detect-sequence-hands-one --source clip.mp4
    gesto image-static-hands-one photo.jpg

Utilities:
    gesto list
    gesto inspect ./gesto_projects/signs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, artifacts
from .regions import REGION_KEYS


def _add_root(p):
    p.add_argument("--root", default=artifacts.DEFAULT_ROOT,
                   help="artifact root folder (default: artifacts)")


def _add_draw_flags(p):
    """--draw / --no-draw. Landmarks are drawn by default."""
    g = p.add_mutually_exclusive_group()
    g.add_argument("--draw", dest="draw", action="store_true", default=True,
                   help="draw landmarks on the frame (default)")
    g.add_argument("--no-draw", dest="draw", action="store_false",
                   help="do not draw landmarks")


def _add_train_args(p):
    p.add_argument("project_dir", help="Gesto project folder (Copy path button)")
    p.add_argument("--seq-len", type=int, default=30,
                   help="frames per sequence, sequence mode (default 30)")
    p.add_argument("--epochs", type=int, default=None,
                   help="training epochs (default 200 static / 300 sequence)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--small", dest="small", action="store_true", default=None,
                   help="force the lighter model")
    p.add_argument("--large", dest="small", action="store_false",
                   help="force the full-size model")
    p.add_argument("--raw", action="store_true",
                   help="data captured with Gesto 'Normalise' unchecked")
    _add_root(p)


def _add_detect_args(p):
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--version", dest="run_version", default=None,
                   help="model version (default: newest)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--smooth", type=int, default=5,
                   help="frames of agreement before committing a label")
    p.add_argument("--width", type=int, default=960, help="display width")
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   default=None, help="don't mirror the webcam")
    _add_draw_flags(p)
    _add_root(p)


def _add_image_args(p):
    p.add_argument("image", help="path to an image file")
    p.add_argument("--version", dest="run_version", default=None,
                   help="model version (default: newest)")
    p.add_argument("--width", type=int, default=960, help="display width")
    p.add_argument("--no-show", dest="show", action="store_false", default=True,
                   help="print the result without opening a window")
    _add_draw_flags(p)
    _add_root(p)


def _region_flag(region):
    return region.replace("_", "-")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gesto",
        description="Train and run gesture models from Gesto Labeller datasets.")
    parser.add_argument("--version", action="version",
                        version=f"gesto {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("mode", choices=artifacts.MODES)
    t.add_argument("region", choices=REGION_KEYS)
    _add_train_args(t)

    d = sub.add_parser("detect", help="run live detection (camera or video)")
    d.add_argument("mode", choices=artifacts.MODES)
    d.add_argument("region", choices=REGION_KEYS)
    _add_detect_args(d)

    im = sub.add_parser("image", help="classify a single image (static models)")
    im.add_argument("region", choices=REGION_KEYS)
    im.set_defaults(mode="static")
    _add_image_args(im)

    # per-combination commands
    for mode in artifacts.MODES:
        for region in REGION_KEYS:
            rflag = _region_flag(region)
            tp = sub.add_parser(f"train-{mode}-{rflag}",
                                help=f"train {mode} model for {region}")
            tp.set_defaults(mode=mode, region=region)
            _add_train_args(tp)
            dp = sub.add_parser(f"detect-{mode}-{rflag}",
                                help=f"detect {mode} {region} (camera or video)")
            dp.set_defaults(mode=mode, region=region)
            _add_detect_args(dp)
    for region in REGION_KEYS:
        ip = sub.add_parser(f"image-static-{_region_flag(region)}",
                            help=f"classify an image with static {region}")
        ip.set_defaults(mode="static", region=region)
        _add_image_args(ip)

    ls = sub.add_parser("list", help="show trained models")
    _add_root(ls)

    ins = sub.add_parser("inspect", help="summarise a Gesto project's data")
    ins.add_argument("project_dir")
    ins.add_argument("--seq-len", type=int, default=30)

    return parser


def _cmd_train(args):
    from .train import train
    train(args.project_dir, args.region, args.mode, root=args.root,
          seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size,
          val_split=args.val_split, small=args.small, normalized=not args.raw)
    tail = "" if args.root == artifacts.DEFAULT_ROOT else f" --root {args.root}"
    print(f"\nDetect with:  gesto detect {args.mode} {args.region}{tail}")
    return 0


def _version_arg(args):
    v = getattr(args, "run_version", None)
    if v is not None and str(v).isdigit():
        return int(v)
    return v


def _cmd_detect(args):
    from .detect import run as run_detect
    run_detect(args.mode, args.region, root=args.root, version=_version_arg(args),
               source=args.source, threshold=args.threshold, smooth=args.smooth,
               width=args.width, mirror=args.mirror, draw_landmarks=args.draw)
    return 0


def _cmd_image(args):
    from .detect import predict_image
    predict_image(getattr(args, "mode", "static"), args.region, args.image,
                  root=args.root, version=_version_arg(args),
                  draw_landmarks=args.draw, show=args.show, width=args.width)
    return 0


def _cmd_list(args):
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
                extra = f", seq_len={meta['seq_len']}" if "seq_len" in meta else ""
                print(f"  {run.name:20} {meta.get('region','?'):10} "
                      f"{meta.get('samples','?')} samples{extra}  [{classes}]")
            except FileNotFoundError:
                print(f"  {run.name:20} (no labels.json)")
    if not found:
        print(f"No trained models under {root}/")
    return 0


def _cmd_inspect(args):
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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command
    if cmd == "train" or cmd.startswith("train-"):
        handler = _cmd_train
    elif cmd == "detect" or cmd.startswith("detect-"):
        handler = _cmd_detect
    elif cmd == "image" or cmd.startswith("image-"):
        handler = _cmd_image
    elif cmd == "list":
        handler = _cmd_list
    elif cmd == "inspect":
        handler = _cmd_inspect
    else:
        parser.error(f"unknown command {cmd}")
        return 2
    try:
        return handler(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
