# No Captions, No Problem: Captionless 3D-CLIP Alignment with Hard Negatives via CLIP Knowledge and LLMs

This repository contains the code for the paper "No Captions, No Problem: Captionless 3D-CLIP Alignment with Hard Negatives via CLIP Knowledge and LLMs".

## Installation
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

Then, install [PointNeXt](https://github.com/user/repo/blob/branch/other_file.md) using the following commands:
```bash
cd models/
git clone --recurse-submodules git@github.com:guochengqian/PointNeXt.git
cd PointNeXt
source update.sh
source install.sh
source update.sh
source install.sh
```
## Data
Download the ShapeNet dataset from the [ULIP 2 repository](https://github.com/salesforce/ULIP/tree/main).

We will release soon the text landmarks and their embeddings.
Together with that, we will release the code to preprocess the data and create the precomputed similarity matrices.

## Training
To train the model, run the following command:
```bash
python train.py --args
```
The complete list of arguments can be found in the [train.py](train.py) file.

## Evaluation
We release the code for the evaluation of the model on Zero-Shot classification and Cross-Modal retrieval.
Download the datasets and the pretrained model (soon to be released) and run the corresponding scripts:
```bash
python zero_shot_retrieval.py --args
python cross_modal_retrieval.py --args
```
The complete list of arguments can be found in the [test_zeroshot.py](test_zeroshot.py) and [test_retrieval.py](test_retrieval.py) files.