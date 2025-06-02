# CUDA_VISIBLE_DEVICES="0" python inference_with_DACLIP.py --input ./dataset/test/lr \
#     --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_stage2_TTA \
#     --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP/models/net_g_50000.pth \
#     --start_idx 0 --end_idx 600


# CUDA_VISIBLE_DEVICES="0" python inference_with_DACLIP.py --input ./dataset/test/lr \
#     --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_stage2_TTA \
#     --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP/models/net_g_50000.pth \
#     --start_idx 600 --end_idx 1200


# CUDA_VISIBLE_DEVICES="0" python inference_with_DACLIP.py --input ./dataset/test/lr \
#     --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_stage2_TTA \
#     --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP/models/net_g_50000.pth \
#     --start_idx 1200 --end_idx 1800


CUDA_VISIBLE_DEVICES="5" python inference_with_DACLIP.py --input ./dataset/test/lr \
    --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_with_DACLIP_modify_TTA_v2_145000 \
    --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP_modify_v2/models/net_g_25000.pth


# CUDA_VISIBLE_DEVICES="0" python inference_with_DACLIP.py --input ./dataset/test/lr \
#     --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_stage2_TTA \
#     --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP/models/net_g_50000.pth \
#     --start_idx 2400 --end_idx 3000

# CUDA_VISIBLE_DEVICES="0" python inference_with_DACLIP.py --input ./dataset/test/lr \
#     --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_stage2_TTA \
#     --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP/models/net_g_50000.pth \
#     --start_idx 3000 --end_idx 3600