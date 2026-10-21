import os
import argparse
import ipdb
from tqdm import tqdm

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

def get_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":

    parse = argparse.ArgumentParser("Run Faster RCNN detection to generate instance detection data!!!")
    parse.add_argument("--faster_rcnn_path", '--f_r_p', type=str,
                        default='/root/fasterrcnn-pytorch-training-pipeline/outputs/training/structural_acco_wico_20241023/best_model.pth',
                        help="path to the Faster RCNN checkpoint")
    parse.add_argument('--device', '--d', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help="which device to run the model on")

    args = parse.parse_args()
    
    # set up model
    device = get_device(args.device)
    checkpoint = torch.load(args.faster_rcnn_path, map_location=device)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(coco_model=False, rpn_post_nms_top_n_test=200, \
                                                                 box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3,
                                                                 num_classes=checkpoint['data']['NC'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print('Begining...')
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    io.mkdir_if_not_exists(data_const.faster_rcnn_boxes, recursive=True)

    fc7_feat_hdf5 =  os.path.join(data_const.faster_rcnn_boxes,'faster_rcnn_fc7.hdf5')
    pool_feat_hdf5 =  os.path.join(data_const.faster_rcnn_boxes,'faster_rcnn_pool.hdf5')
    fc7_feat = h5py.File(fc7_feat_hdf5, 'w')
    pool_feat = h5py.File(pool_feat_hdf5, 'w')

    for ind in tqdm(range(len(anno_list))):
        save_dir = data_const.faster_rcnn_boxes
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = Image.open(os.path.join(data_const.images_dir, anno_list[ind]['image_path_postfix'])).convert('RGB')
        input = torchvision.transforms.functional.to_tensor(image)
        input = input.to(device)

        with torch.no_grad():
            outputs = model([input], save_feat=True)
        
        # save object detection result data
        np.save(os.path.join(save_dir, '{}_boxes.npy'.format(anno_list[ind]['global_id'])), outputs[0]['boxes'].cpu().detach().numpy())
        np.save(os.path.join(save_dir, '{}_scores.npy'.format(anno_list[ind]['global_id'])), outputs[0]['scores'].cpu().detach().numpy())
        nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(anno_list[ind]['global_id']))
        io.dump_json_object(outputs[0]['labels'], nms_keep_indices_path)
        fc7_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['fc7_feat'].cpu().detach().numpy())
        pool_feat.create_dataset(anno_list[ind]['global_id'], data=outputs[0]['pool_feat'].cpu().detach().numpy())

    fc7_feat.close()
    pool_feat.close()
    print('Make detection data successfully!')