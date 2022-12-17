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

    # model_file_path = None

    # try:
    #     model_file_path = download_model(DEFAULT_MODEL_WORKSPACE, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION)
    # except NotFoundError:
    #     app.logger.info(e.message)
    #     return
    # except Exception as e:
    #     app.logger.info(e.message)
    #     return
    
    # app.logger.info(f"Model successfully downloaded: workspace={DEFAULT_MODEL_WORKSPACE}, model={DEFAULT_MODEL_NAME}, version={DEFAULT_MODEL_VERSION}")

    # try:
    #     loaded_model = pickle.load(open(model_file_path, 'rb'))
    # except:
    #     app.logger.info(f"Could not load model from disk: workspace={DEFAULT_MODEL_WORKSPACE}, model={DEFAULT_MODEL_NAME}, version={DEFAULT_MODEL_VERSION}")
    #     return

    # app.logger.info(f"Model successfully loaded: workspace={DEFAULT_MODEL_WORKSPACE}, model={DEFAULT_MODEL_NAME}, version={DEFAULT_MODEL_VERSION}")


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

    try:
        model_file_path = download_model(model_workspace, model_name, model_version)
    except NotFoundError:
        app.logger.info(e.message)
        raise APIError(e.message, 404)
    except Exception as e:
        app.logger.info(e.message)
        raise APIError('Internal server error', 500)
    
    app.logger.info(f"Model successfully downloaded: workspace={model_workspace}, model={model_name}, version={model_version}")

    try:
        loaded_model = pickle.load(open(model_file_path, 'rb'))
    except:
        app.logger.info(f"Could not load model from disk: workspace={model_workspace}, model={model_name}, version={model_version}")
        raise APIError('Internal server error', 500)

    response = f"Model successfully loaded: workspace={model_workspace}, model={model_name}, version={model_version}"
    app.logger.info(response)

    return jsonify(response)  # response must be json serializable!


def download_model(workspace: str, model: str, version: str) -> str:
    file_path = join(MODEL_FOLDER_PATH, f"{workspace}~{model}~{version}.sav")

    if(exists(file_path)):
        return file_path

    comet_api_key = os.environ.get('COMET_API_KEY')

    comet_client = CometClient(comet_api_key)
    comet_client.download_registry_model(workspace, model, version, file_path)

    return file_path


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    if (loaded_model == None):
        message = "No model available for prediction. Use /download_registry_model to load a model"
        app.logger.info(message)
        raise APIError(message, 404)

    response = [0.5]

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
