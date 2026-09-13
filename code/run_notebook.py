"""Execute a notebook's code cells as a plain script (no nbconvert dependency).

Usage: NETMOB_DROP_FEATURES=rain_mm NETMOB_ARTIFACT_DIR=./artifacts_x \
       python run_notebook.py ML_autoencoder_cnn.ipynb
"""
import json, sys

nb = json.load(open(sys.argv[1]))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
exec(compile(src, sys.argv[1], "exec"), {"__name__": "__main__"})
