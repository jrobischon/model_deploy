from flask import Flask, jsonify, request
import requests
import json
import datetime
import uuid

def get_utc_timestamp():
    """
    Get current timestamp in UTC, rounded to 3 milliseconds

    Returns:
        UTC timestamp string
    """

    return datetime.datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')

def get_model_version(api_url):
    """
    Get API model version

    Args:
        api_url (str): Model API URL

    Returns:
        version (str): Model version
    """

    r = requests.get(api_url + '/version')

    assert r.status_code == 200, f"Model version request failed with status code {r.status_code}"

    version = json.loads(r.text)
    return version

def get_model_predictions(api_url, end_point, entity_id=None, data=None):
    """
    Make a request to get predictions from API

    Args:
        api_url (str): Model API URL
        end_point (str): Prediction endpoint (i.e. '/predict', '/predict_proba')
        entity_id (JSON serializable object): Entity identifier
        data (list): List of model input features

    Returns:
        results (dict): Model predictions
    """

    json_data = json.dumps({"entity_id": entity_id, "data": data})

    r = requests.post(api_url + end_point, json=json_data)

    assert r.status_code == 200, \
          f"Post request to '{end_point}' failed with status code {r.status_code}"

    results = json.loads(r.text)
    return results


class ModelAPI(object):

    def __init__(self, model):
        self.model = model

    def deploy(self, host, port, debug=False):
        app = Flask('ml_model')

        @app.route("/version", methods=["GET"])
        def version():
            """
            Returns the self.model version

            Returns:
                (str) model version
            """
            return jsonify(self.model.version)

        @app.route("/predict", methods=["POST"])
        def predict():
            """
            Calls the .predict() method from the self.model object

            Returns:
                result (str): JSON string containing prediction results
            """

            # Get the timestamp of request receipt
            received_ts = get_utc_timestamp()

            if request.method == "POST":

                json_data = request.get_json()

                data_dict = json.loads(json_data)

                result = self.model.predict(data_dict["data"])

                # Append timestamps
                result["pred_id"] = uuid.uuid4()
                result["entity_id"] = data_dict["entity_id"]
                result["receipt_ts"] = received_ts
                result["response_ts"] = get_utc_timestamp()

                return jsonify(result)

        @app.route("/predict_proba", methods=["POST"])
        def predict_proba():
            """
            Calls the .predict() method from the self.model object

            Returns:
                result (str): JSON string containing prediction results
            """

            # Get the timestamp of request receipt
            received_ts = get_utc_timestamp()

            if request.method == "POST":

                json_data = request.get_json()

                data_dict = json.loads(json_data)

                result = self.model.predict_proba(data_dict["data"])

                # Append timestamps
                result["pred_id"] = uuid.uuid4()
                result["entity_id"] = data_dict["entity_id"]
                result["receipt_ts"] = received_ts
                result["response_ts"] = get_utc_timestamp()

                return jsonify(result)


        app.run(debug=debug, host=host, port=port)