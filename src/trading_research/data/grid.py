"""Putting a stream of book updates onto a fixed time grid.

An order book arrives as events, not as samples: a busy instrument updates ten
times a second and a quiet one three times a minute. Any analysis that counts
rows — a lookback, a training window, a holding period — is therefore measuring
different amounts of time on different instruments unless the stream is
resampled first.

That was not a hypothetical. Taking every fiftieth update gave five-second rows
on BTCUSDT and fourteen-second rows on BICOUSDT, so "twenty days of training"
became thirty-three days and "a two-minute horizon" became five, and comparing
the two instruments compared their update rates.

Per day, not per span
---------------------
Resampling the whole range at once builds a continuous grid across days that
were never downloaded and forward-fills a stale book into them — sixteen days
out of a hundred, on the run that caught this. Each day is resampled on its own,
so a gap in the data stays a gap.

Carrying forward
----------------
Within a day, an interval with no update repeats the last one. That is the
honest reading — between updates the book is what it last was — and it is
causal, since a repeated row only ever restates something already observed. The
cost is that a quiet instrument produces rows whose returns are zero, which
dilutes any per-row statistic. That is a true statement about the instrument
rather than an artefact of the method: a book that stands still for a minute at
a time is one where a two-minute horizon contains very little.

Which end of the bin a row is labelled by
-----------------------------------------
A bin covers ``[start, start + seconds)`` and carries the **last** value seen in
it. By default the row is labelled by the bin's start, as pandas does, so the
row labelled 00:00:00 holds the book as it stood at 00:00:04.9. Inside
grid-only computations that is consistent — every row is shifted the same way
— but joined into event time at its label it is a look-ahead of up to one bin:
a decision at 00:00:01 would read a value that did not exist until 00:00:04.9.

``label="right"`` labels the same bin by its end, ``start + seconds``. The
values are identical and every row moves one bin later, so each row's label is
after the moment its value was observed and an as-of join on the label is
causal. Anything that joins grid-derived values into event time — the
market-making simulator does — uses the right label.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

#: Seconds per row. Five is what the rest of the project assumes when it counts
#: rows, and changing it changes the meaning of every window expressed in rows.
GRID_SECONDS = 5

#: Columns a resampled frame must still carry to be usable downstream.
REQUIRED = ("timestamp", "bid_price_0", "ask_price_0")


def to_grid(
    book: pd.DataFrame,
    seconds: int = GRID_SECONDS,
    *,
    label: Literal["left", "right"] = "left",
) -> pd.DataFrame:
    """Resample book updates onto a fixed grid, one day at a time.

    ``label`` names the end of the bin a row is labelled by. ``"left"`` (the
    default, unchanged) is the bin's start, which is up to one bin of
    look-ahead when the grid is joined into event time at its labels;
    ``"right"`` is the bin's end, after every observation the row carries. See
    the module docstring.
    """
    missing = [c for c in REQUIRED if c not in book.columns]
    if missing:
        raise KeyError(f"cannot grid a frame without {missing}")
    if seconds < 1:
        raise ValueError(f"seconds must be at least 1, got {seconds}")
    if label not in ("left", "right"):
        raise ValueError(f"label must be 'left' or 'right', got {label!r}")
    if book.empty:
        raise ValueError("cannot grid an empty frame")

    frame = book.set_index("timestamp").sort_index()
    # Bins are closed on the left under either label, so the two labellings
    # carry the same values and differ only by one bin in their labels.
    parts = [
        day.resample(f"{seconds}s", closed="left", label=label).last().ffill()
        for _, day in frame.groupby(frame.index.date, sort=True)  # type: ignore[attr-defined]
    ]
    grid = pd.concat(parts)
    return grid.dropna(subset=["bid_price_0", "ask_price_0"]).reset_index()
