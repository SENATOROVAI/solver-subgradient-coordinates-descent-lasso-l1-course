"""Run all twenty complete coding answers and retain actual outcomes."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import runpy
import warnings

ROOT = Path(__file__).resolve().parent
report = {"status": "passed", "solutions": {}}
for number in range(1, 21):
    name = f"ch{number:02d}"
    stream = io.StringIO()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with redirect_stdout(stream):
            scope = runpy.run_path(str(
                ROOT / "exercise_solutions" / (name + ".py")))
    result = scope["RESULT"]
    json.dumps(result, allow_nan=False)
    report["solutions"][name] = {
        "result": result, "stdout": stream.getvalue(),
        "warnings": [{"category": item.category.__name__,
                      "message": str(item.message)} for item in seen],
    }
    print(name + " solution completed", flush=True)
(ROOT / "exercise_verification.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2,
               allow_nan=False) + "\n", encoding="utf-8")
