"""Feature engineering. Every transform must be causal: the value at row t may only
use data from rows <= t. tests/test_features.py checks this."""
