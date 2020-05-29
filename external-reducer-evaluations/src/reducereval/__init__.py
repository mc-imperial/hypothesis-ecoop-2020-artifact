import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
assert os.path.exists(os.path.join(ROOT, "src"))

CORPORA = os.path.join(ROOT, "corpora")
WORKING = os.path.join(ROOT, ".working")
