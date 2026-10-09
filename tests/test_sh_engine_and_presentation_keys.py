"""The engine's catalogue and the presentation's key table read every key alike."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting import search_keys
from trading_research.search_engine.catalogue import Catalogue

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"


def test_both_tables_and_neighbourhoods_agree_on_the_registered_catalogue() -> None:
    catalogue = Catalogue.load(CONFIG)
    keys = list(catalogue.keys)
    engine = catalogue.table().reset_index(drop=True)
    shown = search_keys.catalogue_table(keys).reset_index(drop=True)
    assert set(engine.columns) == set(shown.columns)
    for column in shown.columns:
        if column in search_keys.NUMERIC_AXES or column == "regime_h":
            np.testing.assert_array_equal(
                engine[column].to_numpy(float), shown[column].to_numpy(float), err_msg=column
            )
        else:
            pd.testing.assert_series_equal(
                engine[column].astype(str), shown[column].astype(str), check_names=False
            )
    np.testing.assert_array_equal(catalogue.neighbourhoods(), search_keys.neighbourhoods(shown))
    assert search_keys.S25_KEY in catalogue
