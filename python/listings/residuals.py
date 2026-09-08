from pure_core import (_number, _vector)

def residuals(y, prediction):
    """Остаток имеет знак: наблюдение минус прогноз."""
    y = _vector(y, "y")
    prediction = _vector(prediction, "prediction")
    if len(y) != len(prediction):
        raise ValueError("y и prediction: разная длина")
    return [_number(a - b, "Остаток") for a, b in zip(y, prediction)]
