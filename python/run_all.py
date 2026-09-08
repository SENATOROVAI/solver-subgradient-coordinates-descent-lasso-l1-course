"""Run the core checks and all chapter experiments: python run_all.py."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import platform
import runpy
import subprocess
import sys
import time
import warnings

import matplotlib
import numpy
import scipy
import sklearn


ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "test_core.py")], check=True)
report = {
    "status": "passed",
    "environment": {
        "python": platform.python_version(),
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    },
    "chapters": {},
}
for number in range(1, 21):
    name = f"ch{number:02d}"
    output = io.StringIO()
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with redirect_stdout(output):
            scope = runpy.run_path(str(ROOT / "snippets" / (name + ".py")))
    result = scope["RESULT"]
    json.dumps(result, allow_nan=False)
    figures = scope["FIGURES"]
    assert len(figures) >= 2, name
    for stem in figures:
        for suffix in (".pdf", ".png"):
            path = ROOT.parent / "figures" / (stem + suffix)
            assert path.is_file() and path.stat().st_size > 1000, path
    report["chapters"][name] = {
        "result": result, "figures": figures,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout": output.getvalue(),
        "warnings": [{"category": w.category.__name__,
                      "message": str(w.message)} for w in seen],
    }
    print(name + " completed", flush=True)
    for item in seen:
        print("  WARNING: " + str(item.message), flush=True)
report["core"] = json.loads((ROOT / "core_verification.json").read_text())
(ROOT / "verification.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2,
               allow_nan=False) + "\n", encoding="utf-8")
print("All 20 chapters and core checks completed.", flush=True)
