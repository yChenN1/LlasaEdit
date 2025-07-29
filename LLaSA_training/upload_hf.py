from huggingface_hub import HfApi

api = HfApi()
# api.upload_folder(
#     folder_path="/mnt/bn/tanman-yg/chenqi/code/LlasaEdit/LLaSA_training/online/a2a/finetune/0718_a2a_wotext_base_llsa1B_lr1e-4/checkpoint-50000",
#     path_in_repo="0718_a2a_wotext_base_llsa1B_lr1e-4",
#     repo_id="chenqi1126/Llasa_ckpts",
#     repo_type="model"
# )


# api.upload_folder(
#     folder_path="/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0728base_lr1e4",
#     path_in_repo="0728base_lr1e4",
#     repo_id="chenqi1126/Clap_ckpts",
#     repo_type="model"
# )

api.upload_file(
    path_or_fileobj="/mnt/bn/tanman-yg/chenqi/code/StyleClap/exp/0728base_lr1e4/2025_07_28-20_38_01-model_HTSAT-base-lr_0.0001-b_96-j_6-p_fp32/checkpoints/epoch_45.pt",
    path_in_repo="0728base_lr1e4/epoch_45.pt",
    repo_id="chenqi1126/Clap_ckpts",
    repo_type="model",
)