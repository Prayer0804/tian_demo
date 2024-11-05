from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
# xxx
#
# from transformers import AutoModel
#
# access_token = "xxx"
#
# model = AutoModel.from_pretrained("private/model", token=access_token


# ---login---
# from huggingface_hub import login
# login()
# ------
# another way
# from huggingface_hub import  whoami
# user = whoami(token="xxx")

from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="super-cool-model")
# private repository







