CUDA_VISIBLE_DEVICES="1" python inference.py --input ../dataset/test/lr/ \
    --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_TTA_with_CBAM_new_195000/ \
    --model_path /mnt/SSD6/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DCRT_pretrain_with_CBAM/models/net_g_195000.pth