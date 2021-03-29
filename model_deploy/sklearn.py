from joblib import load
import numpy as np
import json
import logging

class SklearnModel(object):
    def __init__(self, filename, version):
        self.filename = filename
        self.version = version
        self.model = None

    def _load_model(self):
        logging.info(f"Loading model {self.filename}")
        self.model = load(self.filename)

    @staticmethod
    def _transform_input(X):
        """
        Convert input from JSON object into a numpy array

        Args:
            X (JSON object) : Input JSON data

        Returns:
            X_ (np.array) : Numpy array
        """

        X_ = json.loads(X)
        X_ = np.array(X_)

        if len(X.shape) == 1:
            # Convert from 1D to 2D array
            X_ = X_.reshape(1, -1)

        return X_


    def predict(self, X):
        """


        :param X:
        :return:
        """

        if self.model is None:
            self._load_model()

        # Transform input into numpy array
        X = self._transform_input(X)

        if hasattr(self.model, 'predict'):
            try:
                y_pred = self.model.predict(X).tolist()
                return_code = 0
                logging.info(
                    f"Making predictions for model version {self.version}"
                    f"Input: {X}"
                    f"Predictions: {y_pred}"
                )
            except ValueError:
                y_pred = None
                return_code = 1

        else:
            logging.error("Model .predict() method does not exist")
            y_pred = None
            return_code = 2

        pred_out = {"predictions": y_pred, "version": self.version}

        return return_code, pred_out


    def predict_proba(self, X):
        """


        :param X:
        :return:
        """

        if self.model is None:
            self._load_model()

        # Transform input into numpy array
        X = self._transform_input(X)

        try:
            y_pred = self.model.predict_proba(X).tolist()
            return_code = 0
            logging.info(
                f"Making predictions with model version {self.version}"
                f"Input: {X}"
                f"Predictions: {y_pred}"
            )
        except ValueError as e:
            logging.error(f"ERROR: Invalid input for model version {self.version}",
                          f"Input: {X}",
                          f"Error message: {e}")
            y_pred = None

        pred_out = {"predictions": y_pred, "version": self.version}

        return return_code, y_pred