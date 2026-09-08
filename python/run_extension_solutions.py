"""Выполнить программные ответы дополнительных практикумов."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import runpy
import sys
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from extra_utils import _json_value

report = {'status': 'passed', 'solutions': {}}
for number in range(1, 27):
    name = f'e{number:02d}'
    stream = io.StringIO()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        with redirect_stdout(stream):
            scope = runpy.run_path(str(
                ROOT / 'extension_solutions' / (name + '.py')))
    result = json.loads(json.dumps(scope['RESULT'],
                                   default=_json_value, allow_nan=False))
    report['solutions'][name] = {
        'result': result, 'stdout': stream.getvalue(),
        'warnings': [{'category': w.category.__name__,
                      'message': str(w.message)} for w in seen]}
    print(name + ' solution completed', flush=True)
(ROOT / 'extension_solution_verification.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8')
