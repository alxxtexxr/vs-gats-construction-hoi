mkdir -p datasets/processed/hico
uv run -m datasets.hico_mat_to_json
uv run -m datasets.hico_hoi_cls_count
uv run -m datasets.hico_split_ids
uv run -m datasets.hico_run_faster_rcnn --faster_rcnn_path='../faster_rcnn.pth' --device='auto'
uv run -m datasets.hico_select_confident_boxes
uv run -m datasets.hico_word2vec
uv run -m datasets.hico_train_val_test_data --feat_type='fc7' 
uv run -m datasets.hico_spatial_feature