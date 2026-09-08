from pure_core import (_predict, _training_data, _vector, mae)

def grid_mae(X, y, w_values, b_values):
    """Полный перебор пар (w, b) для одного признака."""
    X, y = _training_data(X, y)
    if len(X[0]) != 1:
        raise ValueError("grid_mae: нужен ровно один признак")
    w_values = _vector(w_values, "w_values")
    b_values = _vector(b_values, "b_values")
    records = []
    for w in w_values:
        for b in b_values:
            loss = mae(y, _predict(X, [w], b))
            records.append((w, b, loss))
    w, b, loss = min(records, key=lambda record: record[2])
    return {"coef": [w], "intercept": b, "loss": loss,
            "records": records}
