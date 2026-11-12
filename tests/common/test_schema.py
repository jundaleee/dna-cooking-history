import pandas as pd
import pytest

from rne.common import schema


def test_delta_bp():
    assert schema.DELTA_BP == 482


def test_check_columns_raises():
    with pytest.raises(ValueError):
        schema.check_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "x")
