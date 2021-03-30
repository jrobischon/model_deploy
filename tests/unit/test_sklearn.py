import pytest
import numpy as np
import json
from model_deploy import SklearnModel
from sklearn.ensemble import RandomForestClassifier

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


def test_transform_1d_to_numpy():
    # Test that input is converted to numpy
    sk_model = SklearnModel(filename=None, version='0.1.0')

    # Check 1D array is converted to 2D
    x = json.dumps([1, 2, 3])
    x_ = sk_model._transform_input(x)

    np.testing.assert_array_equal(x_, np.array([[1,2,3]]))


def test_transform_2d_to_numpy():
    # Test that input is converted to numpy
    sk_model = SklearnModel(filename=None, version='0.1.0')

    # Check 2D array is converted properly
    x = json.dumps([[1, 2, 3], [4, 5, 6]])
    x_ = sk_model._transform_input(x)

    np.testing.assert_array_equal(x_, np.array([[1,2,3], [4, 5, 6]]))


def test_model_lazy_load(random_forest_binary):
    # Test that model is lazily loaded
    sk_model = SklearnModel(filename=random_forest_binary,
                            version='0.1.0')

    assert sk_model.model is None

    sk_model._load_model()
    assert sk_model.model is not None


def test_model_load_correct(random_forest_binary):
    # Test that model is loaded correctly
    sk_model = SklearnModel(filename=random_forest_binary,
                            version='0.1.0')

    sk_model._load_model()
    assert isinstance(sk_model.model, RandomForestClassifier)
    assert sk_model.model.feature_importances_.shape == (10,)


def test_predict_valid_input(binary_classifier_data, random_forest_binary):

    X, y = binary_classifier_data

    X_json = json.dumps(X.tolist())

    sk_model = SklearnModel(filename=random_forest_binary,
                            version='0.1.0')

    y_pred = sk_model.predict(X_json)

    assert is_jsonable(y_pred)
    assert isinstance(y_pred, dict)
    assert isinstance(y_pred["predictions"], list)
    assert len(y_pred["predictions"]) == len(y)
    assert y_pred["version"] == sk_model.version

def test_predict_proba_valid_input(binary_classifier_data, random_forest_binary):

    X, y = binary_classifier_data

    X_json = json.dumps(X.tolist())

    sk_model = SklearnModel(filename=random_forest_binary,
                            version='0.1.0')

    y_pred = sk_model.predict_proba(X_json)

    assert is_jsonable(y_pred)
    assert isinstance(y_pred, dict)
    assert isinstance(y_pred["predictions"], list)
    assert len(y_pred["predictions"]) == len(y)
    assert len(y_pred["predictions"][0]) == 2
    assert y_pred["version"] == sk_model.version
