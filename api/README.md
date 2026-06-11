---
title: DSFSFT41 - Machine Learning Flood Forecasting API
emoji: 💦
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
short_description: Serve the ml-flood-forecasting project's API.
---

##### Application links

- API server : https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space
- API documentation : https://nicolaspichon35-dsfsft41-ml-flood-forecasting-api.hf.space/docs

##### Sources tracking

###### `api` repository 
- The huggingface process requires that the `api` folder is packed as a Git repositiory.
- As a subfolder, the `api` repository is seen as a project's submodule by the project's repository.
- To define the àpi`repository as a submodule:
    - [...]

###### Pushing

1. Create an Huggingface token with Write access: 
- From "https://huggingface.co/settings/tokens":
    - Create new token > 
        - Write
        - Token name
        - Create token -> hf_xxx
        - Copy the token to clipboard
- Create/open api/.env
- Define the environement variable: `HF_<HF_SPACE_NAME>_WRITE_TOKEN=hf_xxx (the toekn's value)`
- Check that the local .gitignore ignores .env file

2. Push local sources to remote origin:
```
git remote set-url origin https://<hf_user_name>:<hf_token>@huggingface.co/spaces/<hf_user_name>/<hf_space_name>
git push origin main    
```
> Note that the `api` repository's element `.git` is a _file_ (and not a _folder_) due to the fact that 'api` is a nested repository. 
> So the `.git` element do not store the remote repo's url ass it would have done with a normal repo.

- Case where the remote url must be set before each push:
    - On windows:
    ```
    git config --global credential.helper manager
    ```

##### Python environment
- Create a virtual environment for the api's server (can be locally tested with uvicorn)
```
python3 -m venv .venv-ml4ff-api
source .venv-ml4ff-api/bin/activate
pip -r requirements.txt
```
