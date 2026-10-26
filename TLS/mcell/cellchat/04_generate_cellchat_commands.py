#!/usr/bin/env python3
"""Generate one CellChat shell command per sample RDS."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parent
DEFAULT_RDS = ROOT / "seurat_object"
DEFAULT_OUT = ROOT / "cellchat_output"
DEFAULT_CMD = ROOT / "cmd"
DEFAULT_LOG = ROOT / "logs"
DEFAULT_ENV = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/conda/envs/scenv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rds-dir", type=Path, default=DEFAULT_RDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cmd-dir", type=Path, default=DEFAULT_CMD)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--conda-env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--queue", default="general")
    parser.add_argument("--group", default="compute-dinglab")
    parser.add_argument("--memory-mb", type=int, default=128_000)
    parser.add_argument("--walltime", default="24:00")
    parser.add_argument("--docker-image", default="scao/dailybox")
    return parser.parse_args()


def quote(path: Path | str) -> str:
    return shlex.quote(str(path))


def main() -> None:
    args = parse_args()
    args.cmd_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "03_run_cellchat.R"
    rds_paths = sorted(args.rds_dir.glob("*.rds"))
    if not rds_paths:
        raise FileNotFoundError(f"No RDS files found in {args.rds_dir}")

    command_paths = []
    for rds_path in rds_paths:
        command_path = args.cmd_dir / f"{rds_path.stem}.sh"
        text = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'eval "$(conda shell.bash hook)"',
                "export PYTHONNOUSERSITE=1",
                f"conda activate {quote(args.conda_env)}",
                (
                    f"Rscript {quote(runner)} "
                    f"--input {quote(rds_path)} "
                    f"--outdir {quote(args.out_dir)} "
                    f"--min-cells {args.min_cells}"
                ),
                "",
            ]
        )
        command_path.write_text(text)
        command_path.chmod(0o755)
        command_paths.append(command_path)

    submit_list = args.cmd_dir / "commands_to_submit.txt"
    submit_list.write_text(
        "".join(f"bash {quote(path)}\n" for path in command_paths)
    )
    bsub_list = args.cmd_dir / "bsub_commands_to_submit.txt"
    bsub_lines = []
    for path in command_paths:
        sample = path.stem
        bsub_lines.append(
            " ".join(
                [
                    "bsub",
                    "-q",
                    shlex.quote(args.queue),
                    "-n 1",
                    f"-M {args.memory_mb}",
                    (
                        f'-R "span[hosts=1] '
                        f'rusage[mem={args.memory_mb}]"'
                    ),
                    "-G",
                    shlex.quote(args.group),
                    "-a",
                    shlex.quote(f"docker({args.docker_image})"),
                    "-W",
                    shlex.quote(args.walltime),
                    "-J",
                    shlex.quote(f"TLS_CC_{sample}"),
                    "-o",
                    quote(args.log_dir / f"{sample}.%J.out"),
                    "-e",
                    quote(args.log_dir / f"{sample}.%J.err"),
                    "bash",
                    quote(path),
                ]
            )
        )
    bsub_list.write_text("".join(f"{line}\n" for line in bsub_lines))
    print(f"Wrote {len(command_paths)} commands to: {args.cmd_dir}")
    print(f"Command list: {submit_list}")
    print(f"LSF bsub list: {bsub_list}")


if __name__ == "__main__":
    main()
