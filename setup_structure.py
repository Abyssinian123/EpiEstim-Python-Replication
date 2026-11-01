
from pathlib import Path

root = Path(__file__).resolve().parent

folders = [
    "data/raw",
    "data/processed",
    "data/simulated",
    "src/epiestim",
    "experiments/replication",
    "experiments/extension",
    "notebooks",
    "results/figures",
    "results/tables",
    "tests",
    "docs/presentation",
]

files = [
    "requirements.txt",
    "src/epiestim/__init__.py",
    "src/epiestim/renewal.py",
    "src/epiestim/em_algorithm.py",
    "src/epiestim/rt_estimation.py",
    "src/epiestim/simulation.py",
    "src/epiestim/reporting.py",
    "src/epiestim/metrics.py",
    "experiments/replication/reproduce_influenza.py",
    "experiments/replication/reproduce_covid.py",
    "experiments/replication/reproduce_simulation.py",
    "experiments/replication/reproduce_aggregation.py",
    "experiments/extension/constant_reporting.py",
    "experiments/extension/sudden_change.py",
    "experiments/extension/gradual_change.py",
    "experiments/extension/sensitivity_analysis.py",
    "tests/test_em.py",
    "tests/test_rt.py",
    "tests/test_reporting.py",
    "docs/methodology.md",
    "docs/replication_notes.md",
]

for folder in folders:
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)

    # Git 不追踪空目录
    (path / ".gitkeep").touch(exist_ok=True)

for filename in files:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print("Project structure created successfully!")
