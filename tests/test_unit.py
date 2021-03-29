import pytest
import numpy as np
import json


def is_jsonable(x):
    """
    Check if input x can be converted to JSON

    Args:
        x : Any python object

    Returns:
        Boolean, True = Input x can be converted to JSON, False = Input x cannot be converted to JSON
    """
    try:
        _ = json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


@pytest.mark.parametrize('input,expected',
                         [
                            ('test', True),
                            (np.array([1,2,3]), False),
                            ([1,2,3], True),
                            (1, True),
                            (1.23, True),
                            ({'x': [1, 3, 4]}, True)
                         ]
                        )
def test_is_jsonable_correct(input,expected):
    # Check that is_jsonable returns correct results
    assert is_jsonable(input) == expected

