# Scene Consistency Representation Learning for Video Scene Segmentation (CVPR2022)
This is an official PyTorch implementation of SCRL, the CVPR2022 paper is available at [here](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Scene_Consistency_Representation_Learning_for_Video_Scene_Segmentation_CVPR_2022_paper.html).

# Getting Started

## Data Preparation
### MovieNet Dataset 
Download MovieNet Dataset from its [Official Website](https://movienet.github.io/).
### SceneSeg318 Dataset
Download the Annotation of [SceneSeg318](https://drive.google.com/drive/folders/1NFyL_IZvr1mQR3vR63XMYITU7rq9geY_?usp=sharing), you can find the download instructions in [LGSS](https://github.com/AnyiRao/SceneSeg/blob/master/docs/INSTALL.md) repository.

### Make Puzzles for pre-training
In order to reduce the number of IO accesses and perform data augmentation (a.k.a *Scene Agnostic Clip-Shuffling* in the paper) at the same time, we suggest to stitch 16 shots into one image (puzzle) during the pre-training stage. You can make the data by yourself:
```
python ./data/data_preparation.py
```
And the processed data will be saved in `./compressed_shot_images/`, a puzzle example [figure](./figures/puzzle_example.jpg).
<!-- Or download the processed data in [here](). -->


### Load the Data into Memory [Optional]
We **strongly recommend** loading data into memory to speed up pre-training, which additionally requires your device to have at least 100GB of RAM.
```
mkdir /tmpdata
mount tmpfs /tmpdata -t tmpfs -o size=100G
cp -r ./compressed_shot_images/ /tmpdata/
```


## Initialization Weights Preparation
Download the ResNet-50 weights trained on ImageNet-1k ([resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)), and save it in `./pretrain/` folder.

## Prerequisites
### Requirements
* python >= 3.6
* pytorch >= 1.6
* cv2
* pickle
* numpy
* yaml
* sklearn

### Hardware
* 8 NVIDIA V100 (32GB) GPUs

# Usage
### STEP 1: Encoder Pre-training
Using the default configuration to pretrain the model. Make sure the data path is correct and the GPUs are sufficient (e.g. 8 NVIDIA V100 GPUs)
```
python pretrain_main.py --config ./config/SCRL_pretrain_default.yaml
```
The checkpoint, copy of config and log will be saved in `./output/`.

### STEP 2: Feature Extraction

```
python extract_embeddings.py $CKP_PATH --shot_img_path $SHOT_PATH --Type all --gpu-id 0
```
`$CKP_PATH` is the path of an encoder checkpoint, and `$SHOT_PATH` is the keyframe path of MovieNet.
The extracted embeddings (in pickle format) and log will be saved in `./embeddings/`.

### STEP 3: Video Scene Segmentation Evaluation

```
cd SceneSeg

python main.py \
    -train $TRAIN_PKL_PATH \
    -test  $TEST_PKL_PATH \
    -val   $VAL_PKL_PATH \
    --seq-len 40 \
    --gpu-id 0
```

The checkpoints and log will be saved in `./SceneSeg/output/`.

## Models
We provide checkpoints, logs and results under two different pre-training settings, i.e. with and without ImageNet-1K initialization, respectively.

| Initialization | AP | F1 | Config File | STEP 1 <br> Pre-training | STEP 2 <br> Embeddings| STEP 3 <br>  Fine-tuning  |
| :-----| :---- | :---- | :---- | :-----| :---- | :---- |
| w/o  ImageNet-1k | 55.16 | 51.32 | SCRL_pretrain <br> _without_imagenet1k.yaml | [ckp and log](https://drive.google.com/drive/folders/1ZYg9PFRU_lt3G5qJrldkguA52T2oxErR?usp=sharing) | [embedings](https://drive.google.com/drive/folders/1uen_HP3BZu8bcrPBikkgV3j9wzUjQ0C1?usp=sharing) | [log](https://drive.google.com/drive/folders/1rJbOnVbqTdPmnh2grIkePXOmwpNELnrK?usp=sharing) |
| w/ ImageNet-1k | 56.65 | 52.45 | SCRL_pretrain <br> _with_imagenet1k.yaml | [ckp and log](https://drive.google.com/drive/folders/1BG5ZLqrPKKGTtDIZj8aps_QuWc6K3c3V?usp=sharing) | [embedings](https://drive.google.com/drive/folders/1NFvGhkvRxpmEJYNjRnwp3ybuHQaG25gW?usp=sharing) | [log](https://drive.google.com/drive/folders/1dE0JFi-MDua70_CgI1CvyLNRnhwLjaUV?usp=sharing) |


## License
Please see [LICENSE](./LICENSE) file for the details.

## Acknowledgments
Part of codes are borrowed from the following repositories:
* [MoCo](https://github.com/facebookresearch/moco)
* [LGSS](https://github.com/AnyiRao/SceneSeg)

## Citation
Please cite our work if it's useful for your research.
```
@InProceedings{Wu_2022_CVPR,
    author    = {Wu, Haoqian and Chen, Keyu and Luo, Yanan and Qiao, Ruizhi and Ren, Bo and Liu, Haozhe and Xie, Weicheng and Shen, Linlin},
    title     = {Scene Consistency Representation Learning for Video Scene Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14021-14030}
}
```