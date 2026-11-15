"""The firm's live paper books. Changing a book's strategy means a new book name."""
from __future__ import annotations

from datetime import date

from firm.live.runner import BookSpec, LiveContext

PROBATION_START = date(2026, 11, 30)  # first decision at this close, first fills 1 Dec open


def _momplus(ctx: LiveContext):
    from firm.strategies.momentum_plus import MomentumPlus

    # Same settings as the researched config "mp_lgbm_none+res".
    return MomentumPlus(ctx.sectors, ctx.nifty, ctx.vix, picker="lgbm", dial="none",
                        seeds=(0, 1, 2), picker_steps=100_000, dial_steps=60_000, meetings=ctx.meetings)


def _momentum(ctx: LiveContext):
    from firm.strategies.baselines import Momentum

    return Momentum()


def books(start: date = PROBATION_START) -> list[BookSpec]:
    return [
        BookSpec(
            name="swing_probation", team="swing_stocks", status="probation", capital=100_000.0,
            label="Swing: LightGBM + results", agent="Picker (LightGBM)", start=start,
            factory=_momplus, needs_meetings=True,
            note="Probation to 31 May 2027. Promote: return >= equal-weight - 5pts, max DD better "
                 "than -25%, no risk breaches, trades match backtest. Cut: DD past -25% or "
                 "trailing equal-weight by 10pts.",
        ),
        BookSpec(
            name="swing_shadow_momentum", team="swing_stocks", status="shadow", capital=100_000.0,
            label="Shadow: plain momentum", agent="Momentum (shadow)", start=start,
            factory=_momentum, note="Yardstick only, no firm capital.",
        ),
    ]
