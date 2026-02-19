"""CLI entrypoint for xrfv2-edge-tal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from xrfv2_edge_tal.config import apply_cli_overrides, load_yaml_config
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.data.prepare import prepare_dataset
from xrfv2_edge_tal.data.probe import probe_xrfv2_h5
from xrfv2_edge_tal.modalities import resolve_requested_modalities

try:
    import typer

    HAS_TYPER = True
except ModuleNotFoundError:
    typer = None  # type: ignore[assignment]
    HAS_TYPER = False


def _echo(msg: str) -> None:
    print(msg)


def _adapter_from_name(adapter: str, data_root: str, seed: int) -> Any:
    if adapter == "dummy":
        return DummyAdapter(seed=seed)
    if adapter == "xrfv2":
        return XRFV2H5Adapter(data_root)
    raise ValueError(f"Unknown adapter '{adapter}'. Use one of: dummy, xrfv2")


def cmd_download(data_root: str = "data/raw/xrfv2", kaggle_dataset: str = "xrfv2/xrf-v2") -> None:
    out_dir = Path(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_kaggle_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    kaggle_bin = shutil.which("kaggle")

    if has_kaggle_env and kaggle_bin:
        command = [
            kaggle_bin,
            "datasets",
            "download",
            "-d",
            kaggle_dataset,
            "-p",
            str(out_dir),
            "--unzip",
        ]
        try:
            subprocess.run(command, check=True)
            _echo(f"Downloaded dataset into {out_dir}")
            return
        except subprocess.CalledProcessError as exc:
            _echo(f"Kaggle CLI command failed: {exc}")

    _echo("Kaggle download skipped. Provide dataset manually in this structure:")
    _echo(f"  {out_dir}/train_data.h5")
    _echo(f"  {out_dir}/train_label.json")
    _echo(f"  {out_dir}/test_data.h5")
    _echo(f"  {out_dir}/test_label.json")
    _echo(f"  {out_dir}/info.json")


def _directory_summary(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        out.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size_bytes": child.stat().st_size if child.is_file() else None,
            }
        )
    return out


def cmd_inspect(
    adapter: str = "dummy",
    data_root: str = "data/raw/xrfv2",
    seed: int = 42,
    list_modalities: bool = False,
    show_shapes: bool = False,
) -> None:
    root = Path(data_root)
    summary = _directory_summary(root)

    _echo(f"Data root: {root}")
    if summary:
        _echo("Top-level files:")
        for row in summary:
            suffix = f" ({row['size_bytes']} bytes)" if row["size_bytes"] is not None else ""
            _echo(f"  - {row['type']}: {row['name']}{suffix}")
    else:
        _echo("Top-level files: <none>")

    try:
        ds = _adapter_from_name(adapter, data_root=str(root), seed=seed)
    except Exception as exc:
        _echo(f"Adapter inspection failed: {exc}")
        return

    details: dict[str, Any] = {
        "adapter": adapter,
        "modalities": ds.modalities,
        "train_count": len(ds.split_ids("train")),
        "test_count": len(ds.split_ids("test")),
    }
    if list_modalities:
        canonical = resolve_requested_modalities(ds.modalities, requested_modalities=None)
        details["canonical_modalities"] = canonical
    if show_shapes and adapter == "xrfv2":
        details["shape_probe"] = probe_xrfv2_h5(data_root=root, sample_index=0)
    elif show_shapes:
        details["shape_probe"] = {"warning": "--show-shapes currently supports adapter=xrfv2 only"}
    _echo(json.dumps(details, indent=2, sort_keys=True))


def cmd_prepare(
    adapter: str = "dummy",
    data_root: str = "data/raw/xrfv2",
    output_dir: str = "data/processed",
    seed: int = 42,
) -> None:
    ds = _adapter_from_name(adapter, data_root=data_root, seed=seed)
    paths = prepare_dataset(ds, data_root=data_root, output_dir=output_dir, seed=seed)
    _echo("Prepared dataset artifacts:")
    for key, path in paths.items():
        _echo(f"  - {key}: {path}")


def cmd_train(
    config: str,
    data_root: str,
    adapter: str,
    seed: int,
    overrides: list[str],
    runs_dir: str,
) -> None:
    from xrfv2_edge_tal.train import train_main

    cfg = load_yaml_config(config)
    cfg = apply_cli_overrides(cfg, overrides)
    run_dir = train_main(
        config=cfg, data_root=data_root, adapter_name=adapter, seed=seed, runs_dir=runs_dir
    )
    _echo(f"Training run dir: {run_dir}")
    _echo(f"Checkpoint: {run_dir / 'checkpoints' / 'last.npz'}")


def cmd_eval(
    checkpoint: str,
    config: str,
    data_root: str,
    adapter: str,
    seed: int,
    overrides: list[str],
    output_dir: str,
) -> None:
    from xrfv2_edge_tal.eval import eval_main

    cfg = load_yaml_config(config)
    cfg = apply_cli_overrides(cfg, overrides)
    run_dir = eval_main(
        checkpoint=checkpoint,
        config=cfg,
        data_root=data_root,
        adapter_name=adapter,
        seed=seed,
        output_dir=output_dir,
    )
    _echo(f"Eval run dir: {run_dir}")
    _echo(f"Metrics: {run_dir / 'metrics.json'}")


def _apply_profile_modalities(cfg: dict[str, Any], profile: str | None) -> dict[str, Any]:
    cfg_data = dict(cfg.get("data", {}))
    profiles = cfg_data.get("profiles", {})
    default_profile = str(cfg_data.get("default_profile", "earbuds_glasses"))
    selected_profile = profile or default_profile

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            "Config is missing data.profiles. Expected mapping like "
            "{earbuds_glasses: [...], glasses_only: [...]}."
        )
    if selected_profile not in profiles:
        available = ", ".join(sorted(str(k) for k in profiles.keys()))
        raise ValueError(f"Unknown profile '{selected_profile}'. Available profiles: {available}")

    selected_modalities = profiles[selected_profile]
    if not isinstance(selected_modalities, list) or not all(
        isinstance(item, str) for item in selected_modalities
    ):
        raise ValueError(
            f"Invalid data.profiles.{selected_profile}. Expected list[str], got {type(selected_modalities)!r}"
        )

    cfg_data["modalities"] = selected_modalities
    cfg_data["selected_profile"] = selected_profile
    cfg["data"] = cfg_data
    return cfg


def cmd_event_train(
    config: str,
    data_root: str,
    adapter: str,
    seed: int,
    runs_dir: str,
    profile: str | None,
    overrides: list[str],
) -> None:
    from xrfv2_edge_tal.event.train_event import train_event_main

    cfg = load_yaml_config(config)
    cfg = apply_cli_overrides(cfg, overrides)
    cfg = _apply_profile_modalities(cfg, profile)
    run_dir = train_event_main(
        config=cfg,
        data_root=data_root,
        adapter_name=adapter,
        seed=seed,
        runs_dir=runs_dir,
        profile=cfg["data"]["selected_profile"],
    )
    _echo(f"Event training run dir: {run_dir}")
    _echo(f"Profile: {cfg['data']['selected_profile']}")
    _echo(f"Checkpoint: {run_dir / 'checkpoints' / 'last.npz'}")


def cmd_event_eval(
    checkpoint: str,
    config: str,
    data_root: str,
    adapter: str,
    seed: int,
    output_dir: str,
    profile: str | None,
    profiles: str | None,
    overrides: list[str],
) -> None:
    from xrfv2_edge_tal.event.eval_event import eval_event_main

    cfg = load_yaml_config(config)
    cfg = apply_cli_overrides(cfg, overrides)
    profile_list = [item.strip() for item in profiles.split(",")] if profiles else None
    default_profile = profile_list[0] if profile_list else profile
    cfg = _apply_profile_modalities(cfg, default_profile)
    run_dir = eval_event_main(
        checkpoint=checkpoint,
        config=cfg,
        data_root=data_root,
        adapter_name=adapter,
        seed=seed,
        output_dir=output_dir,
        profile=cfg["data"]["selected_profile"],
        profiles=profile_list,
    )
    _echo(f"Event eval run dir: {run_dir}")
    _echo(f"Profile: {cfg['data']['selected_profile']}")
    _echo(f"Metrics: {run_dir / 'metrics.json'}")


def cmd_benchmark(checkpoint: str, config: str, seed: int, output_dir: str) -> None:
    from xrfv2_edge_tal.benchmark import benchmark_main

    cfg = load_yaml_config(config)
    run_dir = benchmark_main(checkpoint=checkpoint, config=cfg, seed=seed, output_dir=output_dir)
    _echo(f"Benchmark run dir: {run_dir}")
    _echo(f"Benchmark: {run_dir / 'benchmark.json'}")


def cmd_export_onnx(checkpoint: str, config: str, output_path: str, seed: int) -> None:
    from xrfv2_edge_tal.export_onnx import export_onnx_main

    cfg = load_yaml_config(config)
    try:
        path = export_onnx_main(
            checkpoint=checkpoint, config=cfg, output_path=output_path, seed=seed
        )
    except RuntimeError as exc:
        _echo(f"ONNX export failed: {exc}")
        return
    _echo(f"ONNX exported: {path}")


if HAS_TYPER:
    app = typer.Typer(help="XRF V2 Edge-First TAL Benchmark CLI")

    @app.command("download")
    def download(
        data_root: str = "data/raw/xrfv2",
        kaggle_dataset: str = "xrfv2/xrf-v2",
    ) -> None:
        cmd_download(data_root=data_root, kaggle_dataset=kaggle_dataset)

    @app.command("inspect")
    def inspect(
        adapter: str = "dummy",
        data_root: str = "data/raw/xrfv2",
        seed: int = 42,
        list_modalities: bool = False,
        show_shapes: bool = False,
    ) -> None:
        cmd_inspect(
            adapter=adapter,
            data_root=data_root,
            seed=seed,
            list_modalities=list_modalities,
            show_shapes=show_shapes,
        )

    @app.command("prepare")
    def prepare(
        adapter: str = "dummy",
        data_root: str = "data/raw/xrfv2",
        output_dir: str = "data/processed",
        seed: int = 42,
    ) -> None:
        cmd_prepare(adapter=adapter, data_root=data_root, output_dir=output_dir, seed=seed)

    @app.command("train")
    def train(
        config: str = typer.Option(..., "--config"),
        data_root: str = "data/raw/xrfv2",
        adapter: str = "dummy",
        seed: int = 42,
        runs_dir: str = "runs",
        override: list[str] | None = None,
    ) -> None:
        cmd_train(
            config=config,
            data_root=data_root,
            adapter=adapter,
            seed=seed,
            overrides=override or [],
            runs_dir=runs_dir,
        )

    @app.command("eval")
    def evaluate(
        checkpoint: str = typer.Option(..., "--checkpoint"),
        config: str = typer.Option(..., "--config"),
        data_root: str = "data/raw/xrfv2",
        adapter: str = "dummy",
        seed: int = 42,
        output_dir: str = "runs",
        override: list[str] | None = None,
    ) -> None:
        cmd_eval(
            checkpoint=checkpoint,
            config=config,
            data_root=data_root,
            adapter=adapter,
            seed=seed,
            overrides=override or [],
            output_dir=output_dir,
        )

    @app.command("event-train")
    def event_train(
        config: str = typer.Option(..., "--config"),
        data_root: str = "data/raw/xrfv2",
        adapter: str = "dummy",
        seed: int = 42,
        runs_dir: str = "runs",
        profile: str | None = None,
        override: list[str] | None = None,
    ) -> None:
        cmd_event_train(
            config=config,
            data_root=data_root,
            adapter=adapter,
            seed=seed,
            runs_dir=runs_dir,
            profile=profile,
            overrides=override or [],
        )

    @app.command("event-eval")
    def event_evaluate(
        checkpoint: str = typer.Option(..., "--checkpoint"),
        config: str = typer.Option(..., "--config"),
        data_root: str = "data/raw/xrfv2",
        adapter: str = "dummy",
        seed: int = 42,
        output_dir: str = "runs",
        profile: str | None = None,
        profiles: str | None = None,
        override: list[str] | None = None,
    ) -> None:
        cmd_event_eval(
            checkpoint=checkpoint,
            config=config,
            data_root=data_root,
            adapter=adapter,
            seed=seed,
            output_dir=output_dir,
            profile=profile,
            profiles=profiles,
            overrides=override or [],
        )

    @app.command("benchmark")
    def benchmark(
        checkpoint: str = typer.Option(..., "--checkpoint"),
        config: str = typer.Option(..., "--config"),
        seed: int = 42,
        output_dir: str = "runs",
    ) -> None:
        cmd_benchmark(checkpoint=checkpoint, config=config, seed=seed, output_dir=output_dir)

    @app.command("export-onnx")
    def export_onnx(
        checkpoint: str = typer.Option(..., "--checkpoint"),
        config: str = typer.Option(..., "--config"),
        output_path: str = typer.Option(..., "--output-path"),
        seed: int = 42,
    ) -> None:
        cmd_export_onnx(checkpoint=checkpoint, config=config, output_path=output_path, seed=seed)

else:

    def app() -> None:
        parser = argparse.ArgumentParser(prog="xrfv2-edge-tal")
        sub = parser.add_subparsers(dest="command", required=True)

        p_download = sub.add_parser("download")
        p_download.add_argument("--data-root", default="data/raw/xrfv2")
        p_download.add_argument("--kaggle-dataset", default="xrfv2/xrf-v2")

        p_inspect = sub.add_parser("inspect")
        p_inspect.add_argument("--adapter", default="dummy", choices=["dummy", "xrfv2"])
        p_inspect.add_argument("--data-root", default="data/raw/xrfv2")
        p_inspect.add_argument("--seed", type=int, default=42)
        p_inspect.add_argument("--list-modalities", action="store_true")
        p_inspect.add_argument("--show-shapes", action="store_true")

        p_prepare = sub.add_parser("prepare")
        p_prepare.add_argument("--adapter", default="dummy", choices=["dummy", "xrfv2"])
        p_prepare.add_argument("--data-root", default="data/raw/xrfv2")
        p_prepare.add_argument("--output-dir", default="data/processed")
        p_prepare.add_argument("--seed", type=int, default=42)

        p_train = sub.add_parser("train")
        p_train.add_argument("--config", required=True)
        p_train.add_argument("--data-root", default="data/raw/xrfv2")
        p_train.add_argument("--adapter", default="dummy", choices=["dummy", "xrfv2"])
        p_train.add_argument("--seed", type=int, default=42)
        p_train.add_argument("--runs-dir", default="runs")
        p_train.add_argument("--override", action="append", default=[])

        p_eval = sub.add_parser("eval")
        p_eval.add_argument("--checkpoint", required=True)
        p_eval.add_argument("--config", required=True)
        p_eval.add_argument("--data-root", default="data/raw/xrfv2")
        p_eval.add_argument("--adapter", default="dummy", choices=["dummy", "xrfv2"])
        p_eval.add_argument("--seed", type=int, default=42)
        p_eval.add_argument("--output-dir", default="runs")
        p_eval.add_argument("--override", action="append", default=[])

        p_event_train = sub.add_parser("event-train")
        p_event_train.add_argument("--config", required=True)
        p_event_train.add_argument("--data-root", default="data/raw/xrfv2")
        p_event_train.add_argument("--adapter", default="dummy", choices=["dummy", "xrfv2"])
        p_event_train.add_argument("--seed", type=int, default=42)
        p_event_train.add_argument("--runs-dir", default="runs")
        p_event_train.add_argument("--profile", default=None)
        p_event_train.add_argument("--override", action="append", default=[])

        p_event_eval = sub.add_parser("event-eval")
        p_event_eval.add_argument("--checkpoint", required=True)
        p_event_eval.add_argument("--config", required=True)
        p_event_eval.add_argument("--data-root", default="data/raw/xrfv2")
        p_event_eval.add_argument("--adapter", default="dummy", choices=["dummy", "xrfv2"])
        p_event_eval.add_argument("--seed", type=int, default=42)
        p_event_eval.add_argument("--output-dir", default="runs")
        p_event_eval.add_argument("--profile", default=None)
        p_event_eval.add_argument("--profiles", default=None)
        p_event_eval.add_argument("--override", action="append", default=[])

        p_bench = sub.add_parser("benchmark")
        p_bench.add_argument("--checkpoint", required=True)
        p_bench.add_argument("--config", required=True)
        p_bench.add_argument("--seed", type=int, default=42)
        p_bench.add_argument("--output-dir", default="runs")

        p_onnx = sub.add_parser("export-onnx")
        p_onnx.add_argument("--checkpoint", required=True)
        p_onnx.add_argument("--config", required=True)
        p_onnx.add_argument("--output-path", required=True)
        p_onnx.add_argument("--seed", type=int, default=42)

        args = parser.parse_args()
        if args.command == "download":
            cmd_download(data_root=args.data_root, kaggle_dataset=args.kaggle_dataset)
        elif args.command == "inspect":
            cmd_inspect(
                adapter=args.adapter,
                data_root=args.data_root,
                seed=args.seed,
                list_modalities=bool(args.list_modalities),
                show_shapes=bool(args.show_shapes),
            )
        elif args.command == "prepare":
            cmd_prepare(
                adapter=args.adapter,
                data_root=args.data_root,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        elif args.command == "train":
            cmd_train(
                config=args.config,
                data_root=args.data_root,
                adapter=args.adapter,
                seed=args.seed,
                overrides=args.override,
                runs_dir=args.runs_dir,
            )
        elif args.command == "eval":
            cmd_eval(
                checkpoint=args.checkpoint,
                config=args.config,
                data_root=args.data_root,
                adapter=args.adapter,
                seed=args.seed,
                overrides=args.override,
                output_dir=args.output_dir,
            )
        elif args.command == "event-train":
            cmd_event_train(
                config=args.config,
                data_root=args.data_root,
                adapter=args.adapter,
                seed=args.seed,
                runs_dir=args.runs_dir,
                profile=args.profile,
                overrides=args.override,
            )
        elif args.command == "event-eval":
            cmd_event_eval(
                checkpoint=args.checkpoint,
                config=args.config,
                data_root=args.data_root,
                adapter=args.adapter,
                seed=args.seed,
                output_dir=args.output_dir,
                profile=args.profile,
                profiles=args.profiles,
                overrides=args.override,
            )
        elif args.command == "benchmark":
            cmd_benchmark(
                checkpoint=args.checkpoint,
                config=args.config,
                seed=args.seed,
                output_dir=args.output_dir,
            )
        elif args.command == "export-onnx":
            cmd_export_onnx(
                checkpoint=args.checkpoint,
                config=args.config,
                output_path=args.output_path,
                seed=args.seed,
            )
        else:
            parser.print_help()
            sys.exit(2)


if __name__ == "__main__":
    app()
