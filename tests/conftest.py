import os
import pytest
from numba.core.serialize import _unpickled_memo


# The code below fixes an issue with numba when running tests.
# The tests would fail when executed in a suite, but would run when they are run alone... The code below fixes this.

# Set threading before any Numba imports happen
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

# Clear on module load
_unpickled_memo.clear()


@pytest.fixture(autouse=True)
def reset_numba_state():
    """Clean up Numba state between tests"""
    _unpickled_memo.clear()
    yield
    _unpickled_memo.clear()
