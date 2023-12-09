# Fusion of U-Net++ and ResNet50 Models for Melanoma Diagnosis from Dermoscopic Images

## Description

This repository presents a fusion model for skin lesion segmentation and classification, tailored for melanoma diagnosis from dermoscopic images. Combining a custom encoder-decoder neural network with a pre-trained classifier, the model achieves an overall accuracy of XX%. To enhance diagnostic precision, the model first extracts the Region of Interest (ROI) from lesion images using a U-Net++ inspired architecture before feeding the samples into a pre-trained ResNet50 model calibrated for binary predictions ('melanoma' or 'non-melanoma'). The workflow is trained and evaluated on the HAM10000 dataset, comprising 10,015 dermoscopic images with corresponding binary masks and gold standard malignant status annotations.

## Getting Started

### Installation
A step-by-step guide on how to install and set up the project:
1. Clone the repository: `git clone https://github.com/KilianZell/CS502_project.git`
2. Download the compressed and assembled HAM10000 dataset available at: https://drive.google.com/file/d/1suJWzU8Oc4yJJraoR6ARsDSo-HFOFNmy/view?usp=share_link
Alternatively, it is possible to reconstitute the dataset by downloading the two image folders and the groundtruth folder from the [Harvard Datavers](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) ('HAM10000_images_part_1.zip', 'HAM10000_images_part_1.zip' and 'HAM10000_segmentations_lesion_tschandl.zip') as well as the groundtruth labels from the [ISIC website](https://challenge.isic-archive.com/data/#2018) with this dowload [link](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip). You will then need to compile the two image folders and rearange the overall folder structure as so:

project-root/
│
├── data/
│   ├── raw/
│   │   └── # Raw data files (e.g., original images, CSV files)
│   ├── processed/
│   │   └── # Processed data generated during pre-processing
│   └── external/
│       └── # External datasets or data from other sources
│
├── notebooks/
│   └── # Jupyter notebooks for exploratory data analysis, prototyping, etc.
│
├── src/
│   ├── preprocessing/
│   │   └── # Code for data pre-processing steps
│   ├── models/
│   │   └── # Code for model architectures, training, and evaluation
│   └── utils/
│       └── # Utility functions and helper scripts
│
├── experiments/
│   └── # Experiment results, model checkpoints, and logs
│
├── docs/
│   └── # Documentation files (e.g., project documentation, guides)
│
├── tests/
│   └── # Unit tests and test datasets
│
├── scripts/
│   └── # Miscellaneous scripts (e.g., data download scripts, setup scripts)
│
├── .gitignore
├── README.md
└── requirements.txt

5. 
4. Download the pre-trained models


### Prerequisites

List any software, libraries, or services that need to be installed before running your project.

