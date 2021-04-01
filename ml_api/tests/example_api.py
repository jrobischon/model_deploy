import pickle
import io
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from ..scikit_model import ScikitModel
from ..flask_api import ModelAPI


def get_model():
    """
    Create RandomForestClassifier pickle IO object

    Returns:
        (io.BytesIO) : Pickle object
    """
    X_train, y_train = make_classification(n_samples=100, n_features=10,
                                           n_classes=2, random_state=80134)

    rf = RandomForestClassifier(random_state=80134)
    rf.fit(X_train, y_train)

    rf_pkl = pickle.dumps(rf)
    return io.BytesIO(rf_pkl)

if __name__ == "__main__":
    # Get model pickle object
    model_file = get_model()

    # Define ScikitModel object
    model = ScikitModel(filename=model_file, version='0.1.0')

    # Initialize and deploy API
    api = ModelAPI(model)
    api.deploy(host='0.0.0.0', port=80)

