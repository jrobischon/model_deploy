import pytest
import pickle
import io
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

@pytest.fixture()
def binary_classifier_data():
    # Create binary classification training data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=80134)
    return X, y

@pytest.fixture()
def multi_classifier_data():
    # Create multi-class classification data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=5, random_state=80134)
    return X, y

@pytest.fixture()
def regression_data():
    # Create regression data
    X, y = make_regression(n_samples=100, n_features=10, random_state=80134)
    return X, y

@pytest.fixture()
def random_forest_binary(binary_classifier_data):
    # Train binary random forest classifier
    X_train, y_train = binary_classifier_data
    rf = RandomForestClassifier(random_state=80134)
    rf.fit(X_train, y_train)

    rf_pkl = pickle.dumps(rf)
    return io.BytesIO(rf_pkl)

@pytest.fixture()
def random_forest_multi(multi_classifier_data):
    # Train multi-class random forest classifier
    X_train, y_train = multi_classifier_data
    rf = RandomForestClassifier(random_state=80134)
    rf.fit(X_train, y_train)

    rf_pkl = pickle.dumps(rf)
    return io.BytesIO(rf_pkl)

@pytest.fixture()
def random_forest_regressor(regression_data):
    # Train random forest regressor
    X_train, y_train = regression_data
    rf = RandomForestRegressor(random_state=80134)
    rf.fit(X_train, y_train)

    rf_pkl = pickle.dumps(rf)
    return io.BytesIO(rf_pkl)

