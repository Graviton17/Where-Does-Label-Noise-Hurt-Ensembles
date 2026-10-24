"""Show the progress of the Phase 5 run (src/run_main.py).

    conda run -n labelnoise python src/progress.py           # show once
    conda run --no-capture-output -n labelnoise python src/progress.py --watch   # refresh every 10 s (Ctrl+C to stop)

The time estimate weights each remaining condition by how long that
dataset/model took in the Phase 4 timing run, so it accounts for the large
datasets (e.g. electricity) coming last.
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MAIN = BASE / "results" / "raw" / "main"
CONDITIONS_PER_MODEL = 13


def snapshot():
    timing = pd.read_csv(BASE / "results" / "raw" / "_timing.csv")
    cost = timing.set_index(["dataset", "model"]).batch_seconds   # relative cost of one condition
    total_cost = cost.sum() * CONDITIONS_PER_MODEL
    total = len(cost) * CONDITIONS_PER_MODEL

    files = list(MAIN.glob("*.csv"))
    done = pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame(columns=["dataset", "model"])
    done_cost = sum(cost.get((d, m), 0) for d, m in zip(done.dataset, done.model))

    log = (MAIN / "progress.log").read_text().splitlines() if (MAIN / "progress.log").exists() else []
    starts = [l for l in log if "START" in l]
    finished = any("DONE" in l for l in log[-1:])
    last = log[-1][10:].strip() if log else "no log yet"

    frac = done_cost / total_cost if total_cost else 0
    width = 40
    bar = "#" * int(width * frac) + "-" * (width - int(width * frac))
    lines = [f"[{bar}] {100 * frac:5.1f} %   ({len(done)} of {total} conditions)"]

    if finished:
        lines.append("Status: FINISHED")
    elif starts and len(done) and frac > 0:
        start = datetime.strptime(starts[-1][:8], "%H:%M:%S").replace(
            year=datetime.now().year, month=datetime.now().month, day=datetime.now().day)
        if start > datetime.now():
            start -= timedelta(days=1)
        elapsed = datetime.now() - start
        # only conditions done in this session count towards the speed estimate
        remaining = elapsed * (1 - frac) / frac if frac < 1 else timedelta(0)
        lines.append(f"Elapsed {str(elapsed).split('.')[0]}   |   about {str(remaining).split('.')[0]} left"
                     f"   |   expected finish ~{(datetime.now() + remaining):%H:%M}")
    lines.append(f"Last: {last}")
    return "\n".join(lines)


PHASE6 = {"rq4": 4 * 10 * 15, "capacity": 6 * 10 * 15, "rq3": 3 * 6 * 3 * 15}   # expected rows


def snapshot6():
    """Phase 6 (src/run_extra.py): one bar per experiment, by number of finished conditions."""
    lines = []
    for exp, total in PHASE6.items():
        files = list((BASE / "results" / "raw" / exp).glob("*.csv"))
        n = sum(len(pd.read_csv(f)) for f in files)
        log = BASE / "results" / "raw" / exp / "progress.log"
        last = log.read_text().splitlines()[-1][10:].strip() if log.exists() else "not started"
        bar = "#" * int(30 * n / total) + "-" * (30 - int(30 * n / total))
        lines.append(f"{exp:9s} [{bar}] {100 * n / total:5.1f} %  ({n}/{total})   last: {last[:70]}")
    return "\n".join(lines)


def snapshot6b():
    """Phase 6B (src/multiclass.py): 20 datasets x 6 models x 10 conditions."""
    total = 20 * 6 * 10
    files = list((BASE / "results" / "raw" / "rq5").glob("*.csv"))
    n = sum(len(pd.read_csv(f)) for f in files)
    log = BASE / "results" / "raw" / "rq5" / "progress.log"
    lines = log.read_text().splitlines() if log.exists() else []
    last = lines[-1][10:].strip() if lines else "not started"
    current = last.split()[0] if lines and "DONE" not in last else "-"
    bar = "#" * int(40 * n / total) + "-" * (40 - int(40 * n / total))
    status = "FINISHED" if lines and "DONE" in lines[-1] else f"running, dataset: {current}"
    return (f"Phase 6B [{bar}] {100 * n / total:5.1f} %  ({n}/{total} conditions, {len(files)}/20 datasets)\n"
            f"Status: {status}")


if __name__ == "__main__":
    show = snapshot6b if "--phase6b" in sys.argv else snapshot6 if "--phase6" in sys.argv else snapshot
    if "--watch" in sys.argv:
        try:
            while True:
                print("\033[2J\033[H" + show(), flush=True)
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        print(show())
