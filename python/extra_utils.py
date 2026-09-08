"""Сохранение результатов дополнительных практикумов."""
import json
from pathlib import Path


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Нельзя сохранить {type(value).__name__} в JSON")


def finish_extra(name, result):
    """Записать фактические числа; не заменяет проверку алгоритма."""
    folder = Path(__file__).resolve().parent / "extra_results"
    folder.mkdir(exist_ok=True)
    target = folder / f"{name}.json"
    data = json.dumps(result, ensure_ascii=False, indent=2,
                      default=_json_value, allow_nan=False)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(data + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"{name}: результаты сохранены в {target.name}")
    return result
