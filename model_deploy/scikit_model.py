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
        """
        Load model from pickle file
        """

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

        if len(X_.shape) == 1:
            # Convert from 1D to 2D array
            X_ = X_.reshape(1, -1)

        return X_


    def predict(self, X):
        """
        Make predictions with model .predict() method

        Args:
            X (JSON object) : Object containing model input features

        Returns:
            pred_out (dict): Dictionary containing model predictions

                Example output:
                    {
                     "predictions" : [1, 0, 1, 0],
                     "version" : '0.1.0'
                     "error_msg" : None
                     }
        """

        if self.model is None:
            self._load_model()

        # Transform input into numpy array
        X = self._transform_input(X)

        if hasattr(self.model, 'predict'):
            try:
                y_pred = self.model.predict(X).tolist()
                err_msg = None
                logging.info(
                    f"Making predictions for model version {self.version}"
                    f"Input: {X}"
                    f"Predictions: {y_pred}"
                )
            except ValueError as e:
                y_pred = None
                err_msg = str(e)
                logging.exception(err_msg)
        else:
            y_pred = None
            err_msg = "Model .predict() method does not exist"
            logging.exception(err_msg)

        pred_out = {"predictions": y_pred, "version": self.version, "error_msg": err_msg}

        return pred_out


    def predict_proba(self, X):
        """
        Make predictions with model .predict_proba() method

        Args:
            X (JSON object) : Object containing model input features

        Returns:
            pred_out (dict): Dictionary containing model predictions

                Example output:
                    {
                     "predictions" : [0.1, 0.22, 0.3, 0.55],
                     "version" : '0.1.0',
                     "error_msg" : None
                     }
        """

        if self.model is None:
            self._load_model()

        # Transform input into numpy array
        X = self._transform_input(X)

        if hasattr(self.model, 'predict_proba'):
            try:
                y_pred = self.model.predict_proba(X).tolist()
                err_msg = None
                logging.info(
                    f"Making predictions with model version {self.version}"
                    f"Input: {X}"
                    f"Predictions: {y_pred}"
                )
            except ValueError as e:
                logging.error(f"ERROR: Invalid input for model version {self.version}"
                              f"Input: {X}"
                              f"Error message: {e}")
                y_pred = None
                err_msg = str(e)
                logging.exception(err_msg)
        else:
            y_pred = None
            err_msg = "Model .predict_proba() method does not exist"
            logging.exception(err_msg)

        pred_out = {"predictions": y_pred, "version": self.version, "error_msg": err_msg}

        return pred_out