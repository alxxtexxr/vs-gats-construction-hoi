from __future__ import print_function
import sys
import os
import numpy as np
import argparse
import ipdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import dgl
import networkx as nx
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw, ImageFont

from model.model import AGRNN
from datasets import metadata
from utils.vis_tool_custom import vis_img_v2 #, vis_img_vcoco 
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn

# from model.vcoco_model import AGRNN as AGRNN_VCOCO
# from datasets.vcoco_constants import VcocoConstants
# from datasets.vcoco_dataset import VcocoDataset
# from datasets.vcoco_dataset import collate_fn as vcoco_collate_fn

from datetime import datetime

# matplotlib.use('TKAgg')

def nms_by_num_confident_actions(boxes, num_confident_actions, iou_thresh=0.5):
    # Convert list of bounding boxes and scores to tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    num_confident_actions_tensor = torch.tensor(num_confident_actions, dtype=torch.float32)  

    # Perform NMS (Non-Maximum Suppression)
    keep_idxs = torch.ops.torchvision.nms(boxes_tensor, num_confident_actions_tensor, iou_thresh)
    return keep_idxs.tolist()

def get_human_action_idxs(node_num, human_num):
    labeled_edge_list = np.cumsum(node_num - np.arange(human_num) - 1)
    labeled_edge_list = np.concatenate(([0], labeled_edge_list[:-1]))

    human_action_idxs = []
    for i in range(human_num):
        created_action_idxs = np.array(labeled_edge_list[:i]) + np.array(range(i-1, -1, -1), dtype=int)
        self_action_idxs = [-1]
        start = labeled_edge_list[i]
        new_action_idxs = np.array(range(start, start+node_num-i-1))
        action_idxs = np.concatenate((created_action_idxs, self_action_idxs, new_action_idxs))
        assert len(action_idxs) == node_num
        human_action_idxs.append(action_idxs)

    return human_action_idxs

def restructure_det_outputs_and_count_confident_actions(node_num, roi_labels, det_outputs, score_thresh=0.001):
    h_idxs = np.where(roi_labels == 1)[0]
    human_num = len(h_idxs)
    human_action_idxs = get_human_action_idxs(node_num, human_num)
    det_outputs_ = np.concatenate((det_outputs, np.zeros(det_outputs.shape[-1]).reshape(1, -1)), axis=0)
    
    h_det_outputs = np.empty((0, node_num, det_outputs.shape[-1]))
    num_confident_actions = np.zeros(node_num)
    
    for h_i in range(human_num):
        h_det_outputs_i = det_outputs_[human_action_idxs[h_i].astype(int)]

        num_confident_actions_i = np.sum(h_det_outputs_i > score_thresh, axis=1)
        num_confident_actions += num_confident_actions_i
        
        h_det_outputs_i = np.expand_dims(h_det_outputs_i, axis=0)
        h_det_outputs = np.concatenate((h_det_outputs, h_det_outputs_i), axis=0)
    
    return h_det_outputs, num_confident_actions

def main(args):
    # Load checkpoint and set up model
    try:
        # use GPU if available else revert to CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
        print("Testing on", device)
        # load checkpoint
        if args.dataset == 'hico':
            checkpoint = torch.load(args.pretrained_hico, map_location=device)
            print('Checkpoint loaded!')
            # set up model and initialize it with uploaded checkpoint
            # ipdb.set_trace()
            data_const = HicoConstants(feat_type=checkpoint['feat_type'])
            model = AGRNN(
                feat_type=checkpoint['feat_type'], 
                bias=checkpoint['bias'], 
                bn=checkpoint['bn'], 
                dropout=checkpoint['dropout'], 
                layer=checkpoint['layers'], 
                diff_edge=checkpoint['diff_edge'],

                # Parameter arguments for ablation study
                multi_attn=checkpoint['multi_attn'], 
                use_spatial_feat=checkpoint['use_spatial_feat'],
                use_feat=checkpoint['use_feat'],
                use_word2vec=checkpoint['use_word2vec'],
            ) 
        # if args.dataset == 'vcoco':
        #     # load checkpoint
        #     checkpoint = torch.load(args.pretrained_vcoco, map_location=device)
        #     data_const = VcocoConstants(feat_type=checkpoint['feat_type'])
        #     model = AGRNN_VCOCO(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        # ipdb.set_trace()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)

    # prepare for data 
    if args.dataset == 'hico':
        # original_imgs_dir = os.path.join(data_const.infer_dir, 'original_imgs/hico')
        original_imgs_dir = './datasets/hico/images/test2024'
        name = f'{args.name}_{args.label_type}_'if args.name else f'{args.label_type}_'
        save_path = os.path.join(data_const.infer_dir, f'processed_imgs/{name}{datetime.now().strftime("%Y%m%d%H%M%S")}')
        print("save_path:", save_path)
        test_dataset = HicoDataset(data_const=data_const, subset='test')
        # original_imgs_dir = './datasets/hico/images/test2015'
        dataloader = sorted(os.listdir(original_imgs_dir))
        # dataloader = ['HICO_test2015_00000128.jpg']
    # else:
    #     original_imgs_dir = os.path.join(data_const.infer_dir, 'original_imgs/vcoco')
    #     # original_imgs_dir = './datasets/vcoco/coco/images/test2014'
    #     save_path = os.path.join(data_const.infer_dir,'processed_imgs/vcoco')
    #     test_dataset = VcocoDataset(data_const=data_const, subset='vcoco_test')
    #     # dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=vcoco_collate_fn)
    #     # dataloader = sorted(os.listdir(original_imgs_dir))
    #     dataloader = ['COCO_val2014_000000150361.jpg']
        
    if not os.path.exists(original_imgs_dir):
        os.makedirs(original_imgs_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('result images will be kept here{}'.format(save_path))

    # data_list = ['HICO_test2015_00000128.jpg']
    for data in tqdm(dataloader):
        # load corresponding data
        # print("Testing on image named {}".format(img))
        if args.dataset == 'hico':
            img = data
            global_id = img.split('.')[0]

            # TEMP
            if global_id not in test_dataset.sub_app_data:
                print(f"glbaol_id {global_id} is skipped!")
                continue

            test_data = test_dataset.sample_date(global_id)
            test_data = collate_fn([test_data])
            det_boxes = test_data['det_boxes'][0]
            # roi_scores = test_data['roi_scores'][0]
            roi_labels = test_data['roi_labels'][0] 
            # edge_labels = test_data['edge_labels']
            node_num = test_data['node_num']
            features = test_data['features'] 
            spatial_feat = test_data['spatial_feat'] 
            if args.label_type == 'fasterrcnn':
                roi_labels_fasterrcnn = test_data['roi_labels_fasterrcnn'][0] 
                word2vec = test_data['word2vec_fasterrcnn']
            else:
                roi_labels_fasterrcnn = None
                word2vec = test_data['word2vec']
        else:
            img = data
            global_id = str(int((data.split('.')[0].split('_')[-1])))
            test_data = test_dataset.sample_date(global_id)
            test_data = vcoco_collate_fn([test_data])
            # img = data['img_name'][0][:].astype(np.uint8).tostring().decode('ascii')
            # test_data = data
            det_boxes = test_data['det_boxes'][0]
            roi_scores = test_data['roi_scores'][0]
            roi_labels = test_data['roi_labels'][0]
            edge_labels = test_data['edge_labels']
            node_num = test_data['node_num']
            features = test_data['features'] 
            spatial_feat = test_data['spatial_feat']
            word2vec = test_data['word2vec']

        # inference
        features, spatial_feat, word2vec = features.to(device), spatial_feat.to(device), word2vec.to(device)
        outputs = model(node_num, features, spatial_feat, word2vec, [roi_labels])    # !NOTE: it is important to set [roi_labels] 
        
        det_outputs = nn.Sigmoid()(outputs)
        det_outputs = det_outputs.cpu().detach().numpy()
        
        # ================================================================
        # FILTER ACTIONS WITH DUPLICATE BOXES
        # ================================================================
        h_det_outputs, num_confident_actions = restructure_det_outputs_and_count_confident_actions(node_num[0], roi_labels, det_outputs, score_thresh=args.thresh)
        keep_idxs = nms_by_num_confident_actions(det_boxes, num_confident_actions, iou_thresh=args.duplicate_iou_thresh)
        # h_keep_idxs = list(set(h_idxs) & set(keep_idxs))
        
        # ================================================================

        # show result
        # import ipdb; ipdb.set_trace()
        if args.dataset == 'hico':
            # CUSTOM
            image_path = os.path.join('datasets/hico/images/test2024', img)
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert('RGB')
            image_temp = image.copy()
            # gt_img, _, _ = vis_img(image, det_boxes, roi_labels, roi_scores, edge_labels.cpu().numpy(), score_thresh=args.thresh)
            # labels = roi_labels_fasterrcnn if args.label_type == 'fasterrcnn' else roi_labels
            
            det_img, det_txt, _ = vis_img_v2(image_temp, det_boxes, 
                                             labels=roi_labels, labels_fasterrcnn=roi_labels_fasterrcnn, label_type=args.label_type, 
                                             raw_action=h_det_outputs, keep_idxs=keep_idxs, score_thresh=args.thresh,
                                             # Filtering parameters
                                             filter_irrelevant_actions=args.filter_irrelevant_actions, 
                                             filter_non_intersecting_det=args.filter_non_intersecting_det, 
                                             filter_distant_det=args.filter_distant_det)
        # if args.dataset == 'vcoco':
        #     image = Image.open(os.path.join(data_const.original_image_dir, 'val2014', img)).convert('RGB')
        #     image_temp = image.copy()
        #     # gt_img = vis_img_vcoco(image, det_boxes, roi_labels, roi_scores, edge_labels.cpu().numpy(), score_thresh=args.thresh)
        #     det_img = vis_img_vcoco(image_temp, det_boxes, roi_labels, roi_scores, det_outputs, score_thresh=args.thresh)
    
        # det_img.save('/home/birl/ml_dl_projects/bigjun/hoi/VS_GATs/inference_imgs/original_imgs'+'/'+img)
        det_img.save(save_path+'/'+img.split("/")[-1])
        with open(save_path+'/'+img.split("/")[-1].split(".")[0]+'.txt', 'w') as f:
            # print("Writing...")
            # Write the string data to the file
            f.write(det_txt)

        # fig = plt.figure(figsize=(100,100))
        # fig.suptitle(img, fontsize=16)
        # ax1 = plt.subplot(1,2,1)
        # ax1.set_title('ground_truth')
        # plt.imshow(np.array(gt_img))
        # plt.axis('off')
        # ax2 = plt.subplot(1,2,2)
        # ax2.set_title('action_detection')
        # plt.imshow(np.array(det_img))
        # plt.axis('off')
        # plt.ion()
        # plt.pause(10)
        # plt.close()
        # ipdb.set_trace()

def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argpfarse.ArgumentTypeError('Boolean value expected!')
        pass

if __name__ == "__main__":
    # set some arguments
    parser = argparse.ArgumentParser(description='inference of the model')

    parser.add_argument('--dataset', type=str, default='hico', choices=['hico', 'vcoco'],
                        help='which datasets you choose: [hico, vcoco, acco]')

    parser.add_argument('--pretrained_hico', '-p_h', type=str, default='./result/hico_checkpoint_trainval.pth', 
                        help='Location of the checkpoint file: ./result/hico_checkpoint_trainval.pth')

    parser.add_argument('--pretrained_vcoco', '-p_v', type=str, default='./checkpoints/vcoco/objthreshold0.4_trainval/epoch_train/checkpoint_600_epoch.pth', 
                        help='Location of the checkpoint file: ./checkpoints/vcoco/v4_640064_mid128_offset_posetohuman_trainval/checkpoint_400_epoch.pth')

    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')
    parser.add_argument('--random_data', type=str2bool, default='false',
                        help='select data randomly from the test dataset: true')
    # CUSTOM
    parser.add_argument('--name', type=str, default='', help="inference name")
    parser.add_argument('--thresh', type=float, default=0.001, help="threshold: 0.001")
    parser.add_argument('--duplicate_iou_thresh', type=float, default=0.5, help="duplicate IoU threshold: 0.5")
    parser.add_argument('--label_type', default='hico', choices=['hico', 'fasterrcnn'], help="label type: [hico, fasterrcnn]")
    parser.add_argument('--filter_irrelevant_actions', '-f_i_a', type=str2bool, default='true', help="filter irrelevant actions: true")
    parser.add_argument('--filter_non_intersecting_det', '-f_n_i_d', type=str2bool, default='true', help="filter non-intersecting detections: true")
    parser.add_argument('--filter_distant_det', '-f_d_d', type=str2bool, default='true', help="filter distant detections: true")
    
    args = parser.parse_args()
    # inferencing
    main(args)