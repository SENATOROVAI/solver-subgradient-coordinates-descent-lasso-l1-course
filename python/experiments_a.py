"""Служебная запись результатов глав 01--10; без скрытого обучения."""
import json
from pathlib import Path


def finish(chapter, result):
    """Сохранить числа, которые использованы в тексте и рисунках."""
    path = Path(__file__).parent / f"results_{chapter:02d}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

