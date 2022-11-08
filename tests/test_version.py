"""
Simple check of library version
"""

import guided_mvs_lib as lib


def test_version():
    assert lib.__version__ == "0.1.0"
