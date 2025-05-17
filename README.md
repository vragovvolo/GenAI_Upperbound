# GenAI_Upperbound

# Setup

The project includes an `.envrc` file with the following contents:

```sh
# filepath: ~/GenAI_Upperbound/.envrc
export UV_PROJECT_ENVIRONMENT=$HOME/.virtualenvs/$(basename "$PWD")
```
Then run the commands:

```sh
uv init . --lib
uv venv --python 3.12
uv sync
source ~/.virtualenvs/GenAI_Upperbound/bin/activate.fish

```