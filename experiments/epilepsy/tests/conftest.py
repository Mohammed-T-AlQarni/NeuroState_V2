import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: `import experiments.epilepsy...`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # test helpers (_fakes.py)
