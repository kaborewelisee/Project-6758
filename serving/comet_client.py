from comet_ml import API
from comet_ml.exceptions import InvalidRestAPIKey, CometRestApiException
from not_found_error import NotFoundError
import json
import os
from os.path import join


class CometClient:
    """
    Api client to communicate with comet server
    """

    def __init__(self, api_key: str) -> None:
        if(api_key is None):
            ValueError("Missing comet api key")
        self.api = API(api_key)

    def download_registry_model(self, workspace: str, registry_name: str, version: str, output_path: str) -> str:
        """
        Download the specific version of the model in the workspace and save it at the location specified by `output_path`

        Params
        - `workspace`: comet workspace
        - `registry_name`: the registry name for the model
        - `version`: the model version
        - `output_path`: file path where to save the downloaded model
        """
        try:
            internal_model_path = './downloading-model'
            self.api.download_registry_model(workspace, registry_name, version, internal_model_path)
            actual_model = os.listdir(internal_model_path)[0]
            os.rename(join(internal_model_path, actual_model), output_path)
            os.rmdir(internal_model_path)
        except InvalidRestAPIKey:
            raise ValueError("Invalid comet api key")
        except CometRestApiException as e:
            if e.response.status_code == 404 or e.response.status_code == 400:
                error = json.loads(e.response.text)
                raise NotFoundError(error['msg'])
            elif e.response.status_code == 403:
                raise ValueError(f"Invalid comet api key")
            else:
                raise e

        return 

