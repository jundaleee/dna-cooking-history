"""원신호 Ct 시험: 기준 염료로 나누지 않은 형광에서 문턱 교차를 잡는다."""
import numpy as np
import pandas as pd

from rne.qpcr.raw_ct import raw_signal_ct


def _mc(ct_true, base=8e4, top=4e5):
    c = np.arange(1, 41, dtype=float)
    rows = []
    for w, ct in ct_true.items():
        y = base + (top / (1 + np.exp(-(c - ct) * 1.2)) if ct else 0 * c)
        rows += [{"well": w, "cycle": ci, "dye": "SYBR", "fluorescence": yi} for ci, yi in zip(c, y)]
        rows += [{"well": w, "cycle": ci, "dye": "ROX", "fluorescence": 3e4} for ci in c]
    return pd.DataFrame(rows)


def test_raw_ct_orders_and_flat_well():
    t = raw_signal_ct(_mc({"A1": 12.0, "A2": 20.0, "A3": None})).set_index("well")
    assert t.loc["A1", "ct_raw"] < t.loc["A2", "ct_raw"]
    assert abs(t.loc["A2", "ct_raw"] - t.loc["A1", "ct_raw"] - 8.0) < 0.2
    assert np.isnan(t.loc["A3", "ct_raw"])
