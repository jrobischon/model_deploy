from flask import Flask, jsonify
import json
import datetime
import uuid

def get_utc_timestamp():
    """
    Get current timestamp in UTC, rounded to 3 milliseconds

    Args:
        None

    Returns:
        UTC timestamp string
    """

    return datetime.datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')


class ModelAPI(object):

    def __init__(self, model):
        self.model = model

    def deploy(self, host, port, debug=False):
        app = Flask('ml_model')

        @app.route("/version", methods=["GET"])
        def version():
            return jsonify(self.model.version)



        @app.route("/predict", methods=["POST"])
        def predict():

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

        app.run(debug=debug, host=host, port=port)