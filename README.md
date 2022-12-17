# IFT-6758

## How to use this repo to create nhl models
1. Clone the repo

2. (Assuming you are using conda) Use `environment.yml` to create your environment, install the required dependencies and activate it
  - New install: <br />
`conda env create -f environment.yml`

  - If environment is already installed but you want to update the dependencies: <br />
`conda env update --file environment.yml --prune`

  - Activate the environment: <br />
`conda activate ift6758-project-conda-env`

3. Create a folder called `data` at the root of your local repo and use it to store all downloaded data and artifacts.
For conveniency, the `data` folder is already excluded from git so you can push with piece of mind.

## How to use the flask service in this repo
1. Create a python3 virtual env using <br />
`python3 -m venv milestone-3-env`

2. Activate the environment using <br />
`source milestone-3-env/bin/activate`

3. Install the requirement if not done yet: need to be done once only, the first time you create the environment
`pip install -r ./ift6758/requirements.txt`
`pip install -e ift6758`

4. Add your comet environment variable
` export COMET_API_KEY=your_api_key_from_comet_portal`

4. Run Flask using your IDE
