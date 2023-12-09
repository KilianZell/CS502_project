# Fusion of U-Net++ and ResNet50 Models for Melanoma Diagnosis from Dermoscopic Images

## Description

This repository presents a fusion model for skin lesion segmentation and classification, tailored for melanoma diagnosis from dermoscopic images. Combining a custom encoder-decoder neural network with a pre-trained classifier, the model achieves an overall accuracy of XX%. To enhance diagnostic precision, the model first extracts the Region of Interest (ROI) from lesion images using a U-Net++ inspired architecture before feeding the samples into a pre-trained ResNet50 model calibrated for binary predictions ('melanoma' or 'non-melanoma'). The workflow is trained and evaluated on the HAM10000 dataset, comprising 10,015 dermoscopic images with corresponding binary masks and gold standard malignant status annotations.

## Getting Started

### Installation

A step-by-step guide on how to install and set up the project.
1. Clone the repository: `git clone https://github.com/KilianZell/CS502_project.git`
2. Change into the project directory: `cd your-project`
3. Install dependencies: `npm install`


### Prerequisites

List any software, libraries, or services that need to be installed before running your project.

