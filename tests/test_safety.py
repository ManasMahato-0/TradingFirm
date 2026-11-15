"""Guard: paper mode must not contain any code path that places real orders."""
from pathlib import Path

FORBIDDEN = ["place_order", "place_basket_orders", "modify_order", "cancel_order",
             "exit_positions", "convert_position"]


def test_no_real_order_calls_in_firm_package():
    pkg = Path(__file__).resolve().parent.parent / "firm"
    hits = [
        f"{p.name}: {name}"
        for p in pkg.rglob("*.py")
        for name in FORBIDDEN
        if name in p.read_text(encoding="utf-8")
    ]
    assert not hits, f"real-order API names found: {hits}"
