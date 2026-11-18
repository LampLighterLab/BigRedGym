# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Batch 7 on MoLab: fine-tune the two identified policies under actuator
limits, evaluate them, and download the runs and the results.

Policies (go2osc_config.FOLLOWUP_BATCHES[7]), each fine-tuned from its
committed checkpoint at 95% of the actuator torque and speed limits with
500 Hz physics:
    deploy_robust_clip_ft2         from deploy_robust seed 2
    even2.5_deploy_track_clip_ft2  from even2.5_deploy_track seed 2 (candidate)

Split over two notebooks with PART (0: the first policy, 1: the second;
None: both here). Run the cells top to bottom. Rerunning skips finished runs
and evaluations.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import subprocess
    import textwrap
    import zipfile
    from pathlib import Path

    import marimo as mo

    return Path, mo, os, re, subprocess, textwrap, zipfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Batch 7: fine-tune under actuator limits

    1. **Setup**: clone or update the repo and install the dependencies.
    2. **Settings**: pick `PART` (0 or 1 to split over two notebooks).
    3. **Train**: 5 seeds x 1000 iterations per policy.
    4. **Evaluate**: grid, power, friction 0.3, pushes, start/stop.
    5. **Download**: the trained runs and the results (two zips).
    """)
    return


@app.cell
def _(Path, os):
    WORK = Path(os.environ.get("QGYM_WORKDIR") or Path.home() / "qgym")
    REPO = WORK / "QGym"
    WARP_CACHE = WORK / "warp_cache"
    WARP_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ["PATH"] = f"{Path.home() / '.local/bin'}:{os.environ['PATH']}"
    os.environ["WARP_CACHE_PATH"] = str(WARP_CACHE)
    os.environ["MPLBACKEND"] = "Agg"

    # MoLab's own venv variables would hijack the repo's uv commands; scrub them.
    _shadowing = (
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONEXECUTABLE",
    )
    ENV = {
        k: v
        for k, v in os.environ.items()
        if k not in _shadowing and not (k.startswith("UV_") and k != "UV_CACHE_DIR")
    }
    ENV["UV_PYTHON"] = "3.11"
    print(f"repo: {REPO}")
    return ENV, REPO, WORK


@app.cell
def _():
    BRANCH = "tripedal_test"
    REPO_URL = "https://github.com/kandicelu7/BigRedGym.git"
    return BRANCH, REPO_URL


@app.cell
def _(BRANCH, ENV, REPO, REPO_URL, subprocess, textwrap):
    _script = textwrap.dedent(f"""
        set -euo pipefail
        command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
        if [ -d {REPO}/.git ]; then
            git -C {REPO} fetch origin {BRANCH}
            git -C {REPO} checkout {BRANCH}
            git -C {REPO} pull --ff-only
        else
            git clone --branch {BRANCH} {REPO_URL} {REPO}
        fi
        cd {REPO}
        git log --oneline -1
        uv python install 3.11
        uv sync --frozen --extra gpu --no-dev
    """)
    subprocess.run(["bash", "-c", _script], check=True, env=ENV)
    print("\nsetup done")
    return


@app.cell
def _():
    # --- edit, then run --------------------------------------------------------
    # PART: 0 or 1 splits the two policies over two notebooks; None runs both.
    PART = None
    SEEDS = 5
    MAX_ITERATIONS = 1000
    DEVICE = "cuda:0"
    NUM_ENVS = 4096
    EVAL_JOBS = 2  # evaluations at once on the GPU

    _all = ["deploy_robust_clip_ft2", "even2.5_deploy_track_clip_ft2"]
    SETTINGS = _all if PART is None else [_all[PART]]
    TASKS = [f"go2osc_wide_{s}" for s in SETTINGS]
    QUEUE = [(t, k) for k in range(SEEDS) for t in TASKS]
    print(f"part {PART}: {SETTINGS}, {len(QUEUE)} runs of {MAX_ITERATIONS} iterations")
    return DEVICE, EVAL_JOBS, MAX_ITERATIONS, NUM_ENVS, QUEUE, SETTINGS, TASKS


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train

    About 25-40 min per run (500 Hz physics). Time the first run before
    leaving it; rerun this cell after an interruption to continue.
    """)
    return


@app.cell
def _(DEVICE, ENV, MAX_ITERATIONS, NUM_ENVS, QUEUE, REPO, re, subprocess):
    import time as _time

    def _done(task, seed):
        return any(
            (REPO / "logs" / task).glob(f"*_seed{seed}/model_{MAX_ITERATIONS}.pt")
        )

    for _i, (_task, _seed) in enumerate(QUEUE, 1):
        if _done(_task, _seed):
            print(f"skip {_task} seed{_seed}: already trained", flush=True)
            continue
        print(f"\n=== [{_i}/{len(QUEUE)}] {_task} seed{_seed}", flush=True)
        _start = _time.monotonic()
        _cmd = [
            "uv", "run", "--frozen", "--extra", "gpu", "--no-dev", "scripts/train.py",
            "--task", _task, "--seed", str(_seed), "--run_name", f"seed{_seed}",
            "--device", DEVICE, "--headless", "--disable_wandb",
            "--num_envs", str(int(NUM_ENVS)),
            "--max_iterations", str(int(MAX_ITERATIONS)),
        ]  # fmt: skip
        _proc = subprocess.Popen(
            _cmd,
            cwd=REPO,
            env={**ENV, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for _line in _proc.stdout:
            _m = re.search(r"Iteration (\d+)/(\d+)", _line)
            if _m and int(_m.group(1)) % 100 == 0:
                print(f"  iteration {_m.group(1)}/{_m.group(2)}", flush=True)
            elif "Initialized from" in _line or "Error" in _line or "Traceback" in _line:
                print(_line, end="", flush=True)
        if _proc.wait() != 0:
            print(f"!!! {_task} seed{_seed} failed with code {_proc.returncode}")
        print(f"  {(_time.monotonic() - _start) / 60:.1f} min", flush=True)
    print("\ntraining finished")
    TRAIN_DONE = True
    return (TRAIN_DONE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate

    Every trained run on this machine is evaluated (evaluations already done
    are skipped): the speed x frequency grid, short power and friction-0.3
    passes, standardized pushes and the start/stop test.
    """)
    return


@app.cell
def _(DEVICE, ENV, EVAL_JOBS, REPO, SETTINGS, TRAIN_DONE, subprocess):
    assert TRAIN_DONE

    def _run(args):
        cmd = ["uv", "run", "--frozen", "--extra", "gpu", "--no-dev", *args]
        print("\n$ " + " ".join(args), flush=True)
        r = subprocess.run(cmd, cwd=REPO, env=ENV, text=True, capture_output=True)
        print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
        if r.returncode:
            print(f"!!! failed with code {r.returncode}")

    _gpu = ["--device", DEVICE, "--jobs", str(EVAL_JOBS)]
    _short = ["--robots", "16", "--settle", "6", "--measure", "6"]
    _run(["scripts/study_eval.py", "--wide", *_gpu])
    _run(["scripts/study_eval.py", "--wide", *_gpu, *_short, "--results", "power_wide"])
    _run(["scripts/study_eval.py", "--wide", *_gpu, *_short, "--friction", "0.3",
          "--results", "friction0.3_wide"])  # fmt: skip
    _run(["scripts/push_eval.py", "--study", "--wide", *_gpu])
    _run(["scripts/startstop_eval.py", "--study", "--settings", *SETTINGS, *_gpu])
    print("\nevaluation finished")
    EVAL_DONE = True
    return (EVAL_DONE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Download

    Two zips: the trained runs (final checkpoint, saved configs and code,
    paths `logs/<task>/<run>/...`) and the evaluation results (paths
    `logs/study_results/...` and `results/study/runs/...`). Unzip both into
    the repo on your PC.
    """)
    return


@app.cell
def _(EVAL_DONE, MAX_ITERATIONS, PART, REPO, TASKS, WORK, mo, zipfile):
    assert EVAL_DONE
    _tag = "both" if PART is None else f"part{PART}"
    _out = WORK / "exports"
    _out.mkdir(parents=True, exist_ok=True)

    _runs_zip = _out / f"batch7_runs_{_tag}.zip"
    with zipfile.ZipFile(_runs_zip, "w", compression=zipfile.ZIP_STORED) as _zf:
        for _task in TASKS:
            for _run in sorted((REPO / "logs" / _task).glob("*_seed*")):
                for _f in sorted(_run.rglob("*")):
                    _keep = not _f.name.startswith("model_") or (
                        _f.name == f"model_{MAX_ITERATIONS}.pt"
                    )
                    if _f.is_file() and _keep:
                        _zf.write(_f, _f.relative_to(REPO))
                print(f"runs: {_task}/{_run.name}")

    _res_zip = _out / f"batch7_results_{_tag}.zip"
    _n = 0
    with zipfile.ZipFile(_res_zip, "w", compression=zipfile.ZIP_DEFLATED) as _zf:
        _roots = [REPO / "logs" / "study_results", REPO / "results" / "study" / "runs"]
        for _root in _roots:
            for _f in sorted(_root.rglob("*")):
                if _f.is_file() and any(f"{t}__" in _f.name for t in TASKS):
                    _zf.write(_f, _f.relative_to(REPO))
                    _n += 1
    print(f"\n{_runs_zip.name}: {_runs_zip.stat().st_size / 1e6:.1f} MB")
    print(f"{_res_zip.name}: {_n} files, {_res_zip.stat().st_size / 1e6:.1f} MB")
    mo.vstack(
        [
            mo.download(
                data=lambda p=_runs_zip: p.read_bytes(),
                filename=_runs_zip.name,
                label="Download trained runs",
            ),
            mo.download(
                data=lambda p=_res_zip: p.read_bytes(),
                filename=_res_zip.name,
                label="Download evaluation results",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
