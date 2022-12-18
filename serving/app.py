"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from os.path import exists
from os.path import join
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from api_error import APIError
from os.path import exists
from os.path import join
from comet_client import CometClient
from not_found_error import NotFoundError
import pickle


import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_FOLDER_PATH = './models'
DEFAULT_MODEL_WORKSPACE = 'ift6758-22-milestone-2'
DEFAULT_MODEL_NAME = 'question-6-random-forest-classifier-base'
DEFAULT_MODEL_VERSION = '2.0.0'

loaded_model = None


app = Flask(__name__)


@app.errorhandler(APIError)
def invalid_api_usage(e: APIError):
    return jsonify(e.to_dict()), e.status_code


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    load_default_model()


def load_default_model():
    """
    Load default model. Download it if not yet on the server
    """
    model_file_path = None
    already_downloaded = False

    try:
        model_file_path, already_downloaded = download_model(DEFAULT_MODEL_WORKSPACE, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION)
    except NotFoundError as e:
        app.logger.info(str(e))
        return
    except Exception as e:
        app.logger.info(str(e))
        return
    
    download_message = "Model successfully downloaded"
    if(already_downloaded):
        download_message = "Model already present on server"

    app.logger.info(f"{download_message}: workspace={DEFAULT_MODEL_WORKSPACE}, model={DEFAULT_MODEL_NAME}, version={DEFAULT_MODEL_VERSION}")

    global loaded_model
    try:
        loaded_model = pickle.load(open(model_file_path, 'rb'))
    except:
        app.logger.info(f"Could not load model from disk: workspace={DEFAULT_MODEL_WORKSPACE}, model={DEFAULT_MODEL_NAME}, version={DEFAULT_MODEL_VERSION}")
        return

    app.logger.info(f"Model successfully loaded: workspace={DEFAULT_MODEL_WORKSPACE}, model={DEFAULT_MODEL_NAME}, version={DEFAULT_MODEL_VERSION}")


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data

    response = []

    if exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                response = list(f)
        except:
            raise APIError("Internal server error", 500)

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded
    if('workspace' not in json):
        raise APIError('Missing workspace param')

    if('model' not in json):
        raise APIError('Missing model param')
    
    if('version' not in json):
        raise APIError('Missing version param')

    model_workspace = json['workspace']
    model_name = json['model']
    model_version = json['version']

    model_file_path = None
    already_downloaded = False

    try:
        model_file_path, already_downloaded = download_model(model_workspace, model_name, model_version)
    except NotFoundError as e:
        app.logger.info(str(e))
        raise APIError(str(e), 404)
    except Exception as e:
        app.logger.info(str(e))
        raise APIError('Internal server error', 500)

    download_message = "Model successfully downloaded"
    if(already_downloaded):
        download_message = "Model already present on server"
    
    app.logger.info(f"{download_message}: workspace={model_workspace}, model={model_name}, version={model_version}")

    global loaded_model
    try:
        loaded_model = pickle.load(open(model_file_path, 'rb'))
    except:
        app.logger.info(f"Could not load model from disk: workspace={model_workspace}, model={model_name}, version={model_version}")
        raise APIError('Internal server error', 500)

    response = f"Model successfully loaded: workspace={model_workspace}, model={model_name}, version={model_version}"
    app.logger.info(response)

    return jsonify(response)  # response must be json serializable!


def download_model(workspace: str, model: str, version: str) -> tuple[str, bool]:
    """
    Download a specific version of the model in a comet workspace

    Params
    - `workspace`: comet workspace
    - `model`: the registry model name
    - `version`: the model version

    Returns a tuple (str, bool)
    - 1st arg: the file path where the model has been downloaded
    - 2nd arg: indicates if the model was already downloaded
    """
    file_path = join(MODEL_FOLDER_PATH, f"{workspace}~{model}~{version}.sav")

    if(exists(file_path)):
        return file_path, True

    comet_api_key = os.environ.get('COMET_API_KEY')

    comet_client = CometClient(comet_api_key)
    comet_client.download_registry_model(workspace, model, version, file_path)

    return file_path, False


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    model_input = None

    try:
        raw_model_input = get_model_params(json)
        model_input = pd.DataFrame.from_dict({ k: [v] for k, v in raw_model_input.items() })
    except ValueError as e:
        app.logger.info(str(e))
        raise APIError(str(e), 400)
    except Exception as e:
        app.logger.info(str(e))
        raise APIError(str(e), 500)

    global loaded_model

    if (loaded_model == None):
        message = "No model available for prediction. Use /download_registry_model to load a model"
        app.logger.info(message)
        raise APIError(message, 404)

    prediction = 0
    prediction_proba = 0 
    try:
        prediction = loaded_model.predict(model_input)[0]
        prediction_proba = loaded_model.predict_proba(model_input)[0][prediction]
    except Exception as e:
        app.logger.info(f"Error while calling predict on model: {str(e)}")
        raise APIError("Internal server error", 500)

    #Return:
    # - The prediction class: 0 or 1
    # - The prediction probability: if prediction == 0, it is the probability of the class 0. If prediction == 1, it is the probability of class 1
    response = { 'prediction': int(prediction), 'prediction_prabability': prediction_proba }
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


MODEL_PARAMS_TYPE = {
    'coordinates_x': float,
    'coordinates_y': float,
    'period': int,
    'game_period_seconds': float,
    'game_elapsed_time': float,
    'shot_distance': float,
    'shot_angle': float,
    'hand_based_shot_angle': float,
    'empty_net': int,
    'last_coordinates_x': float,
    'last_coordinates_y': float,
    'time_since_last_event': float,
    'distance_from_last_event': float,
    'rebond': int,
    'speed_from_last_event': float,
    'shot_angle_change': float,
    'ShotType_Backhand': int,
    'ShotType_Deflected': int,
    'ShotType_Slap Shot': int,
    'ShotType_Snap Shot': int,
    'ShotType_Tip-In': int,
    'ShotType_Wrap-around': int,
    'ShotType_Wrist Shot': int
}

def get_model_params(input_params: dict[str, object]) -> dict[str, object]:
    """
    Get model params from `input_params`

    Params
    - `input_params`: The input where to extract the model params

    Returns
    - `dict[str, object]`: The params required by the model

    Throws
    - `ValueError`: when a param required by the model is missing or is of incorrect type
    """
    result = {}
    for model_param_key, model_param_type in MODEL_PARAMS_TYPE.items():
        if (model_param_key not in input_params):
            raise ValueError(f"Missing model param '{model_param_key}")
        try:
            param_value = model_param_type(input_params[model_param_key])
            result[model_param_key] = param_value
        except ValueError:
            raise ValueError(f"Model param '{model_param_key} should be of type '{model_param_type}'")
    return result
