import os
import ipdb
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import utils.io as io
from datasets.hico_constants import HicoConstants
import h5py
import json

import torchvision
import torch
# from utils.vis_tool import vis_img

# from utils.vis_tool import vis_img
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from torchvision import transforms as transforms


def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)


if __name__ == "__main__":
    # set up model
    best_path = '/root/fasterrcnn-pytorch-training-pipeline/outputs/training/structural_acco_wico_20241023/best_model.pth'
    checkpoint = torch.load(best_path, map_location='cuda')

    NUM_CLASSES = checkpoint['data']['NC']
    CLASSES = checkpoint['data']['CLASSES']
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(coco_model=False,
                                                                 rpn_post_nms_top_n_test=200,
                                                                 box_batch_size_per_image=128,
                                                                 box_score_thresh=0.1, 
                                                                 box_nms_thresh=0.3,
                                                                 num_classes=NUM_CLASSES)
    
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES) 
    model.load_state_dict(checkpoint['model_state_dict'])

    # DEVICE = torch.device('cpu')
    DEVICE = 'cuda:0'
    model.to(DEVICE).eval()

    print('Begining...')
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    io.mkdir_if_not_exists(data_const.faster_rcnn_boxes, recursive=True)

    fc7_feat_hdf5 =  os.path.join(data_const.faster_rcnn_boxes,'faster_rcnn_fc7.hdf5')
    pool_feat_hdf5 =  os.path.join(data_const.faster_rcnn_boxes,'faster_rcnn_pool.hdf5')
    fc7_feat = h5py.File(fc7_feat_hdf5, 'w')
    pool_feat = h5py.File(pool_feat_hdf5, 'w')

    skipped_orig_image_paths = []
    for ind in tqdm(range(len(anno_list))):
        # if ind <= 28930: continue
        
        save_dir = data_const.faster_rcnn_boxes
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        root = '/root/vs-gats-hhi/datasets/hico/images'

        orig_image_path = os.path.join(root, anno_list[ind]['image_path_postfix'])
        
        # Predict like original
        # image = Image.open(os.path.join(root, anno_list[ind]['image_path_postfix'])).convert('RGB')
        # input = torchvision.transforms.functional.to_tensor(image)
        # input = input.to(DEVICE)
        # outputs = model([input], save_feat=True)

        # Predict like /root/fasterrcnn-pytorch-training-pipeline/inference.py
        # orig_image = cv2.imread(orig_image_path)

        # if orig_image is None:
        #     # print(f"{anno_list[ind]=}")
        #     print(f"{orig_image_path} is skipped because orig_image is None")
        #     skipped_orig_image_paths.append(orig_image_path)
        #     continue

        # frame_height, frame_width, _ = orig_image.shape
        # RESIZE_TO = frame_width
        # # orig_image = image.copy()
        # image_resized = resize(orig_image, RESIZE_TO, square=None)
        # image = image_resized.copy()
        # # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = infer_transforms(image)
        # # Add batch dimension.
        # image = torch.unsqueeze(image, 0)

        # with torch.no_grad():
        #     outputs = model(image.to(DEVICE), save_feat=True)

        # Predict like /root/fasterrcnn_based_hoilabel/inference_extract_image_dataset_HHI_2.ipynb
        image = Image.open(orig_image_path)
        input = torchvision.transforms.functional.to_tensor(image)
        input = input.to(DEVICE)

        with torch.no_grad():
            outputs = model([input], save_feat=True)
        
        # save object detection result data
        np.save(os.path.join(save_dir, '{}_boxes.npy'.format(anno_list[ind]['global_id'])), outputs[0]['boxes'].cpu().detach().numpy())
        np.save(os.path.join(save_dir, '{}_scores.npy'.format(anno_list[ind]['global_id'])), outputs[0]['scores'].cpu().detach().numpy())
        nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(anno_list[ind]['global_id']))
        
        # Maps Human 
        # 'human_engineer',    #-- 1 --> human
        # 'human_squating_engineer',    #-- 6 --> human
        # 'human_helper',    #-- 2 --> human
        # 'human_squating_helper',    #-- 7 --> human
        # 'human_squating_master',    #-- 8 --> human
        # 'human_master',    #-- 3 --> human
        # 'human_welder',    #-- 12 --> human
        # 'human_squating_welder',    #-- 9 --> human
        # 'human_worker',    #-- 13 --> human
        # 'human_squating_worker',    #-- 10 --> human
        
        io.dump_json_object(outputs[0]['labels'], nms_keep_indices_path)
        fc7_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['fc7_feat'].cpu().detach().numpy())
        pool_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['pool_feat'].cpu().detach().numpy())

    fc7_feat.close()
    pool_feat.close()
    print('Make detection data successfully!')