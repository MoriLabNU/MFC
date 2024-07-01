# MFC: Masked Frequency Consistency

#  Introduction

This repository contains the PyTorch implementation of:

Masked Frequency Consistency for Domain-Adaptive Semantic Segmentation of Laparoscopic Images, MICCAI 2023.
[[URL]](https://doi.org/10.1007/978-3-031-43907-0_63) [[PDF]](https://rdcu.be/dnwdK)

[//]: # (and its extension [URL&#40;TBA&#41;].)

## Environment Setup

For this project, we used python 3.7.9. We recommend setting up a new virtual
environment:

```shell
conda create -n mfc python==3.7.9
conda activate mfc
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt
```

[//]: # (Further, please download the MiT weights from SegFormer using the)

[//]: # (following script. If problems occur with the automatic download, please follow)

[//]: # (the instructions for a manual download within the script.)

[//]: # ()
[//]: # (```shell)

[//]: # (sh tools/download_checkpoints.sh)

[//]: # (```)

As [DAFormer](https://github.com/lhoyer/DAFormer),
please download the [MiT weights(mit_b5.pth)](https://drive.google.com/drive/folders/19o1AP-RtlrA7IDUgVUkaK5XBw6NqQ--H?usp=drive_link)
pretrained on ImageNet-1K provided by the official
[SegFormer repository](https://github.com/NVlabs/SegFormer) and put them in a
folder `pretrained/` within this project. Only mit_b5.pth is necessary.

## Datasets Setup

A brief instruction on how to set up the Datasets is provided below.
More detailed instruction will be provided later.

**Simulated Dataset:** [Dataset_sim.md](docs/Dataset_sim.md)

**I2I Dataset:** [Dataset_i2i.md](docs/Dataset_i2i.md)

**Cholec Datasets:** [Dataset_cholec.md](docs/Dataset_cholec.md)

The final folder structure should look like this:

```none
MFC
├── ...
├── MFC_DP
├── DATA
│   ├── cholec
│   │   ├── img
│   │   │   ├── train
│   │   │   ├── test
│   │   ├── gt
│   │   │   ├── test
│   ├── simulated
│   │   ├── images
│   │   ├── labels
│   ├── i2i
│   │   ├── images
│   │   ├── labels
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
cd MFC_DP
python tools/convert_datasets/cs8k.py ../DATA/cholec
python tools/convert_datasets/i2i.py ../DATA/i2i
python tools/convert_datasets/i2i.py ../DATA/simulated
```

## Training

A training job can be launched using:

```shell
python run_experiments.py --config configs/mfc_seg/xxx.py
```

The logs and checkpoints are stored in `work_dirs/`.

## Evaluation

A trained model can be evaluated using:

```shell
sh test.sh work_dirs/local-segmentation/run_name
```

The predictions are saved for inspection to
`work_dirs/run_name/preds`
and the mIoU of the model is printed to the console.

## Checkpoints

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

## Acknowledgements

MFC is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [HRDA](https://github.com/lhoyer/HRDA)
* [MIC](https://github.com/lhoyer/MIC)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

[//]: # (* [Monocular-Depth-Estimation-Toolbox]&#40;https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox&#41;)

## Contact

If you have any questions, please contact [Xinkai Zhao](mailto:xkzhao@mori.m.is.nagoya-u.ac.jp).
