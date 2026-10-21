# VS-GATs for Human-Object Interaction (HOI) Detection on Construction Sites

This project is based on the [original VS-GATs repository](https://github.com/birlrobotics/vs-gats).

## Setup
Set up the Python virtual environment and install the project dependencies:
```bash
uv sync
```

## Data Preparation

### Download Data
1. Download the dataset and put it into `datasets/hico`.

2. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `datasets/word2vec`.

### Process Data
1. Copy `utils/generalized_rcnn.py` and `utils/roi_heads.py` to the folder including the implemented Faster R-CNN (`.venv/lib/python3.12/site-packages/torchvision/models/detection/`). We recommend you to backup the origial code first.

2. Start processing by running:
```bash
bash datasets/hico_process.sh
```