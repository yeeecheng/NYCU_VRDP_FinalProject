# NYCU_VRDP_FinalProject Group 9 (Method 2 - DACLIP)

## NOTE: You can change the branch `feature/drct-cbam` to go to the method 1 - CBAM


### Installation
```
git clone https://github.com/yeeecheng/NYCU_VRDP_FinalProject.git
conda create --name drct python=3.8 -y
conda activate drct
git checkout main
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop
```


## How to train the model

You can get tje DRCT pre-trained weight in this [link](https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U)

First,
```
cd options/train
```
You can see a lot of .yml files, which are the training setting.
In the method 2, I use the **train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP.yml**
Then, you will see below command in kaggle_train.sh
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4321 drct/train.py -opt options/train/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP_modify.yml --launcher pytorch
```
You can change the CUDA_VISIBLE_DEVICES, -opt is the .yml file.

Finally,
```
bash kaggle_train.sh
```


## How to inference the model

First, you can get the weight in the folder **experiments**, chose the best weight you have.
Then, you will see below command in kaggle_inference.sh
```
CUDA_VISIBLE_DEVICES="5" python inference_with_DACLIP.py --input ./dataset/test/lr \
    --output ./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_with_DACLIP_modify_TTA_v2_145000 \
    --model_path /swim-pool/yicheng/NYCU_VRDP_FinalProject/experiments/train_DRCT_SRx4_finetune_from_DRCT_pre-train_with_DACLIP_modify_v2/models/net_g_25000.pth
```
You can change the CUDA_VISIBLE_DEVICES, --input is folder you put the lr images, --output folder you want to put the restord images, --model_path your weight.

Then, run
```
bash kaggle_inference.sh
```

After you finished this step, you can generate the submission file of kaggle competition by kaggle_submisison.py
```
python kaggle_submission.py
```
Remember to change the folder path in line 90
```
prediction(
        sample_submission="./dataset/sample_submission.csv",
        lr_folder='./kaggle_test_HR_result/pre-trained_with_kaggle_dataset_with_DACLIP_modify_TTA_v2_145000',
        output_file=f'kaggle_solution.csv'
    )
```



## You also can follow the DRCT tutorial.

### How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Validation data can be download at [this page](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 drct/train.py -opt options/train/train_DRCT_SRx2_from_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.

### How To Inference on your own Dataset?

```
python inference.py --input_dir [input_dir ] --output_dir [input_dir ]  --model_path[model_path]
```


### How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- Then run the following codes (taking `DRCT_SRx4_ImageNet-pretrain.pth` as an example):
```
python drct/test.py -opt options/test/DRCT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.

- Refer to `./options/test/DRCT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/DRCT_tile_example.yml`.**


**Real DRCT GAN SRx4. (Updated)**

| Model | Training Data | Checkpoint | Log |
|:-----------:|:---------:|:-------:|:--------:|
| [Real-DRCT-GAN_MSE_Model](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [DF2K + OST300](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost/code)  | [Checkpoint](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) |  [Log](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link) |
| [Real-DRCT-GAN_Finetuned from MSE](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [DF2K + OST300](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost/code)  |  [Checkpoint](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing)  |  [Log](https://drive.google.com/file/d/15aBV-FFi7I4esUb1vzRmrjMccc5cEEY4/view?usp=drive_link) |

## Citations

If our work is helpful to your reaearch, please kindly cite our work. Thank!

#### BibTeX
    @misc{hsu2024drct,
      title={DRCT: Saving Image Super-resolution away from Information Bottleneck},
      author = {Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
      year={2024},
      eprint={2404.00722},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    @InProceedings{Hsu_2024_CVPR,
      author    = {Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
      title     = {DRCT: Saving Image Super-Resolution Away from Information Bottleneck},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month     = {June},
      year      = {2024},
      pages     = {6133-6142}
    }

## Thanks
A part of our work has been facilitated by [HAT](https://github.com/XPixelGroup/HAT), [SwinIR](https://github.com/JingyunLiang/SwinIR), [LAM](https://github.com/XPixelGroup/X-Low-level-Interpretation) framework, and we are grateful for their outstanding contributions.

A part of our work are contributed by @zelenooki87, thanks for your big contributions and suggestions!

Special thanks to [Phhofm](https://github.com/Phhofm) for providing the 4xRealWebPhoto_v4_drct-l model, which has significantly enhanced our image processing capabilities. The model is available at [Phhofm/models](https://github.com/Phhofm/models/releases/tag/4xRealWebPhoto_v4_drct-l).

## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.
