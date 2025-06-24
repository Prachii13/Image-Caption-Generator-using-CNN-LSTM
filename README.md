# 🧠 Image Caption Generator

## Description
This project generates captions for images using a CNN (InceptionV3) for feature extraction and an LSTM for sequence generation.

## Dataset
Use the Flickr8k or Flickr30k dataset. Place the images in `data/` and the caption file (`Flickr8k.token.txt`) in the same folder.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Generate Caption
```bash
python generate_caption.py <path_to_image>
```

## Author
Prachi Singh

