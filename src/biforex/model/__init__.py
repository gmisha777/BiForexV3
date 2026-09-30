"""Model training, walk-forward validation and evaluation."""

# Model classes: index in predict_proba output <-> triple-barrier label
CLASSES = (-1, 0, 1)  # down, flat, up
PROBA_COLUMNS = ("p_down", "p_flat", "p_up")
