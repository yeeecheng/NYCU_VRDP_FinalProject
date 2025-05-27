# DRCT: Saving Image Super-resolution away from Information Bottleneck (CVPR NTIRE Oral Presentation)






## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8 and **1.12** !!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
```
git clone https://github.com/ming053l/DRCT.git
conda create --name drct python=3.8 -y
conda activate drct
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop
```
## How To Inference on your own Dataset?

```
python inference.py --input_dir [input_dir ] --output_dir [input_dir ]  --model_path[model_path]
```


## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- Then run the following codes (taking `DRCT_SRx4_ImageNet-pretrain.pth` as an example):
```
python drct/test.py -opt options/test/DRCT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/DRCT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/DRCT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Validation data can be download at [this page](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 drct/train.py -opt options/train/train_DRCT_SRx2_from_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.



### [[Paper Link]](https://arxiv.org/abs/2404.00722) [[Project Page]](https://allproj002.github.io/drct.github.io/) [[Poster]](https://drive.google.com/file/d/1zR9wSwqCryLeKVkJfTuoQILKiQdf_Vdz/view?usp=sharing)


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
