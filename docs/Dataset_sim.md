# Simulated Dataset Setup

A brief instruction on how to setup the Simulated Dataset is provided below.
More detailed instruction will be provided later.

## 1. Download the dataset

Please download the [Input Images](http://opencas.dkfz.de/image2image/data/inputs.7z) 
and [Labels: Segmentation](http://opencas.dkfz.de/image2image/data/labels.7z) from 
the [Image2Image](http://opencas.dkfz.de/image2image/) website. 

## 2. Unzip the dataset

Please unzip the downloaded files.

## 3. Prepare the dataset

Please run the following script to prepare the dataset:

```shell
python MFC_DP/tools/convert_datasets/preparation_simulated.py
```

Please change the path in the script to the path of the dataset.

## 4. Move the dataset

Please move the dataset to the folder `DATA/simulated` if necessary.