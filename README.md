# VS-GATs for Human–Object Interaction (HOI) Detection on Construction Sites

This project is based on the [original VS-GATs repository](https://github.com/birlrobotics/vs-gats).

## Requirements
- CUDA ≥ 12.4 (the primary setup targets Linux; CPU is also supported on other platforms)

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python and dependency management. If `pip` is installed, you can install `uv` by running:
    ```bash
    pip install uv
    ```

## Setup
Create the Python virtual environment and install the project dependencies by running:
```bash
uv sync
```

## Data Preparation

### Download Data
1. Download the [dataset](#) and put it into `datasets/hico/`.

2. Download our [Faster R-CNN checkpoint](#) and put it into `dataset/`.  Ensure that the checkpoint is saved as `dataset/faster_rcnn.pth`.

3. Download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and put it into `datasets/word2vec/`.

### Process Data
1. Copy `utils/generalized_rcnn.py` and `utils/roi_heads.py` into the installed `torchvision` detection module directory: `.venv/lib/python3.12/site-packages/torchvision/models/detection/`. We recommend backing up the original files before replacing them.

2. Run the following command to process the data:
```bash
bash datasets/hico_process.sh
```

## Model Training
Start training by running:
```bash
uv run hico_trainval.py \
    --e_v='vs_gats_train' \
    --t_m='epoch' \
    --b_s=32 \
    --f_t='fc7' \
    --layers=1 \
    --lr=0.00001 \
    --drop_prob=0.3 \
    --bias='true' \
    --optim='adam' \
    --bn=False \
    --m_a='false' \
    --d_a='false' \
    --diff_edge='false' \
    --use_feat='true' \
    --use_word2vec='true' \
    --use_spatial_feat='true'
```

You can visualize the training process using TensorBoard:
```bash
uv run tensorboard --logdir=log/
```

Model checkpoints are saved in the `checkpoints/` directory.