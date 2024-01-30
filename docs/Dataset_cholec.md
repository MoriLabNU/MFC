# Cholec80 and CholecSeg8k Dataset Setup

A brief instruction on how to setup the Cholec80 and CholecSeg8k Dataset is provided below.
More detailed instruction will be provided later.

## CholecSeg8k Dataset for testing

### 1. Download the dataset

Please download the CholecSeg8k Dataset from [https://www.kaggle.com/datasets/newslab/cholecseg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k).

### 2. Unzip the dataset

Please unzip the downloaded files.

### 3. Prepare the dataset

Please run the following script to prepare the dataset:

```shell
python MFC_DP/tools/convert_datasets/preparation_cs8k.py
```

Please change the path in the script to the path of the dataset.

### 4. Move the dataset

Please move the dataset to the folder `DATA/cheloc` if necessary.

## Cholec80 Dataset for training

### 1. Download the dataset

Please download the Cholec80 Dataset from [http://camma.u-strasbg.fr/datasets](http://camma.u-strasbg.fr/datasets).

### 2. Unzip the dataset

Please unzip the downloaded files.

### 3. Prepare the dataset

Please run the following script to prepare the dataset:

```shell
python MFC_DP/tools/convert_datasets/preparation_cheloc.py
```

Please change the path in the script to the path of the dataset.

### 4. Move the dataset

Please move the dataset to the folder `DATA/cheloc` if necessary.