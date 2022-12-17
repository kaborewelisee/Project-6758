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
import pickle

MODEL_FOLDER_PATH = './models'

def download_model(workspace: str, model: str, version: str) -> str:
    file_path = join(MODEL_FOLDER_PATH, f"{workspace}~{model}~{version}.sav")

    if(exists(file_path)):
        return file_path

    comet_api_key = os.environ.get('COMET_API_KEY')

    # if(comet_api_key is None):
    #     raise ValueError("Missing COMET_API_KEY")

    comet_client = CometClient("8O5bBL1uMSvxg3JfCPugBMwTM-s")
    comet_client.download_registry_model(workspace, model, version, file_path)
    
    return file_path

if __name__ == "__main__":
    path = download_model("ift6758-22-milestone-2", 'question-6-random-forest-classifier-base', '1.0.0')