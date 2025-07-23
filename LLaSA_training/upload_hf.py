from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/LLaSA_training/online/a2a/finetune/0716_a2a_wotext_base_llsa1B/checkpoint-3200",
    repo_id="chenqi1126/Llasa_ckpts",
    repo_type="model"
)