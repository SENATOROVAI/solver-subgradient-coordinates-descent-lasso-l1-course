"""Выполнить новые практикумы и проверить созданные рисунки."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import runpy
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from extra_utils import _json_value

report = {'status': 'passed', 'lessons': {}}
for number in range(1, 27):
    name = f'e{number:02d}'
    stream = io.StringIO()
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        with redirect_stdout(stream):
            scope = runpy.run_path(str(ROOT / 'extensions' / (name + '.py')))
    result = json.loads(json.dumps(scope['RESULT'],
                                   default=_json_value, allow_nan=False))
    assert len(scope['FIGURES']) >= 2
    for stem in scope['FIGURES']:
        for suffix in ('.pdf', '.png'):
            target = ROOT.parent / 'figures' / (stem + suffix)
            assert target.is_file() and target.stat().st_size > 1000
    report['lessons'][name] = {
        'result': result, 'figures': scope['FIGURES'],
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'stdout': stream.getvalue(),
        'warnings': [{'category': w.category.__name__,
                      'message': str(w.message)} for w in seen]}
    print(name + ' completed', flush=True)
    for w in seen:
        print('  WARNING: ' + str(w.message), flush=True)
(ROOT / 'extension_verification.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
    encoding='utf-8')
