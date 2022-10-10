# IFT-6758

## How to use this repo
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
