from pure_core import (_dot, _number, _scale_stats, _vector)

def unscale_coefficients(w, b, stats):
    w = _vector(w, "w")
    b = _number(b, "b")
    mean, scale = _scale_stats(stats, len(w))
    original_w = [_number(v / s, "w") for v, s in zip(w, scale)]
    original_b = _number(b - _dot(mean, original_w), "b")
    return original_w, original_b
