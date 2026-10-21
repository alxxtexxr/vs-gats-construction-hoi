import os
import argparse
import ipdb
from tqdm import tqdm

import numpy as np
import scipy.io as scio
import pickle
import h5py

from datasets import metadata
from datasets.hico_constants import HicoConstants
import utils.io as io 
from utils.vis_tool import vis_img, vis_img_frcnn
from utils.bbox_utils import compute_iou
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import tkinter

import torchvision
import torch

# matplotlib.use('TKAgg')

# def get_intersection(box1, box2):
#     # !NOTE:this is a bug if box1 have no intersetion with box2
#     return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))

# def compute_area(box):
#     return (box[2]-box[0])*(box[3]-box[1])

def get_node_index(classname, bbox, det_classes, det_boxes, node_num, labeled=True):
    # print("bbox =", bbox)
    # print("det_boxes =", det_boxes)
    bbox = np.array(bbox, dtype=np.float32)
 
    # max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou = 0.05
    max_iou_index = -1
    
    # custom
    total_iou_eq_0 = 0
    total_iou_gt_0 = 0

    # print(f"{classname}", end=": ")
    for i_node in range(node_num):
        if labeled:
            if classname == metadata.coco_classes[det_classes[i_node]]:
                # check bbox overlap
                # intersection_area = compute_area(get_intersection(bbox, det_boxes[i_node, :]))
                # iou = intersection_area/(compute_area(bbox)+compute_area(det_boxes[i_node, :])-intersection_area)
                iou = compute_iou(bbox, det_boxes[i_node,:])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_index = i_node
        else:
            iou = compute_iou(bbox, det_boxes[i_node,:])
            
            # custom
            if iou > 0.1:
                # print(f"{iou:.3f}", end=", ")
                total_iou_gt_0 += 1
            else:
                total_iou_eq_0 += 1
                
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i_node
                # print('iou:', iou, ';', 'index:', i_node, ';', 'bbox:', bbox)
    
    # custom
    # print()
    # print(f"total iou > 0 = {total_iou_gt_0}, total iou == 0 = {total_iou_eq_0}")
    # print("[RETURN VALUE] max_iou_index =", max_iou_index)
    # print()
    
    return max_iou_index

def parse_data(data_const,args):

    assert os.path.exists(data_const.clean_dir), 'Please check the path to annotion file!'

    anno_data = scio.loadmat(data_const.clean_dir + '/anno_bbox.mat')
    print('Load original data successfully!')

    boxes_scores_rpn_ids_labels = h5py.File(data_const.boxes_scores_rpn_ids_labels, 'r')
    print('Load selected instance detection data successfully!')

    if args.feat_type == 'fc7':
        boxes_feat = h5py.File(data_const.faster_det_fc7_feat, 'r')
    else:
        boxes_feat = h5py.File(data_const.faster_det_pool_feat, 'r')
    
    action_class_num = len(metadata.action_classes)
    list_action = anno_data['list_action']

    # to save images with bad selected detection
    bad_dets_imgs = {'0': [], '1': [], 'no_human': []} 
    # parsing data
    for phase in ['bbox_train', 'bbox_test']:

        if not args.vis_result:
            if phase == 'bbox_train':
                if args.feat_type == 'fc7':
                    print('Creating hico_trainval_data_fc7_edge.hdf5 file ...')
                    hdf5_file = os.path.join(data_const.proc_dir,'hico_trainval_data_fc7_edge.hdf5')
                    save_data = h5py.File(hdf5_file,'w')
                else:
                    print('Creating hico_trainval_data_pool_edge.hdf5 file ...')
                    hdf5_file = os.path.join(data_const.proc_dir,'hico_trainval_data_pool_edge.hdf5')
                    save_data = h5py.File(hdf5_file,'w')
            else:
                if args.feat_type == 'fc7':
                    print('Creating hico_test_data_fc7_edge.hdf5 file ...')
                    hdf5_file = os.path.join(data_const.proc_dir,'hico_test_data_fc7_edge.hdf5')
                    save_data = h5py.File(hdf5_file,'w')
                else:
                    print('Creating hico_test_data_pool_edge.hdf5 file ...')
                    hdf5_file = os.path.join(data_const.proc_dir,'hico_test_data_pool_edge.hdf5')
                    save_data = h5py.File(hdf5_file,'w')

        data = anno_data[phase]
        if args.vis_result:
            # img_list = [1761,23,44,50,53,72,75,79,93,109,127,129,138,139,490,496]
            img_list = [14663] #range(data.shape[1])
        else:
            img_list = range(data.shape[1])

        for i, i_img in enumerate(tqdm(img_list)):
            # load detection data
            # ipdb.set_trace()
            img_name = data['filename'][0,i_img][0]
            global_id = img_name.split(".")[0]
            
            # # DEBUG
            # if global_id not in ['ACCO_train2024_00002421']:
            #     continue

            # print("global_id:", global_id)
            
            # custom: skip nonexistent data
            # if global_id not in boxes_scores_rpn_ids_labels:
            #     print("Skipped! global_id is not in boxes_scores_rpn_ids_labels")
            #     continue
            
            selected_det_data = boxes_scores_rpn_ids_labels[global_id]['boxes_scores_rpn_ids']
            det_feat = boxes_feat[global_id]

            if selected_det_data.shape[0] == 0:
                bad_dets_imgs['0'].append(global_id)
                continue
            if selected_det_data.shape[0] == 1:
                bad_dets_imgs['1'].append(global_id)
                continue
            # TypeError: Indexing elements must be in increasing order
            # det_feat = det_feat[selected_det_data[:, 5], :]
            feat = []
            for rpn_id in selected_det_data[:, 5]:
                feat.append(np.expand_dims(det_feat[rpn_id, :], 0))
            try:
                feat = np.concatenate(feat, axis=0)
            except Exception as e:
                ipdb.set_trace()
            det_boxes = selected_det_data[:, :4]
            det_class = selected_det_data[:, -1].astype(int)
            det_scores = selected_det_data[:, 4]

            if len(np.where(det_class == 1)[0])==0:
                print('bad dets:', det_class)
                print('bad global_id:', global_id)
                bad_dets_imgs['no_human'].append(global_id)
                continue

            # from coco label to hico label
            # det_class = np.array(metadata.coco_to_hico)[det_class].astype(int)
            
            # from coco_pytorch(91) label to coco_81 label
            # det_class = np.array(metadata.coco_pytorch_to_coco)[det_class.astype(int)].astype(int)

            # calculate the number of nodes
            human_num = len(np.where(det_class == 1)[0])
            object_num = len(np.where(det_class != 1)[0]) # it's not used in this code, but used just for debugging purpose
            node_num = len(det_class)
            labeled_edge_list = np.cumsum(node_num - np.arange(human_num) - 1)
            labeled_edge_num = labeled_edge_list[-1]
            labeled_edge_list[-1] = 0
            # import ipdb; ipdb.set_trace()
            # parse the original data to get node labels
            edge_labels = np.zeros((labeled_edge_num, action_class_num))
            # ipdb.set_trace()
            if args.vis_result:
                image_gt = Image.open(os.path.join(data_const.clean_dir, 'images/train2015', img_name)).convert('RGB')
                raw_action = np.zeros(117)
            
            # CUSTOM
            # print("================================================")
            # print(f"global_id = {global_id}")
            # print("================================================")

            # print(f"data['hoi'][0,i_img].shape[1] = {data['hoi'][0,i_img].shape[1]}")
            if not data['hoi'][0,i_img].shape[1]:
                print(f"i_img = {i_img} is skipped")
                print()
                continue
            # print()

            # DEBUG
            # if global_id not in [
            #     # 'ACCO_train2024_00000001',
            #     'ACCO_train2024_00000003',
            # ]:
            #     # print("Skipped for debugging...")
            #     continue
            print("global_id:", global_id)

            for i_hoi in range(data['hoi'][0,i_img]['id'].shape[1]):
                try:
                    hoi_id = data['hoi'][0, i_img]['id'][0, i_hoi][0, 0]
                    action_id = metadata.hoi_to_action[hoi_id - 1]  # !NOTE: Need to subtract 1 
                    print("================================")
                    print("hoi id:", hoi_id)
                    print("================================")
                    # print('action_id:', action_id)
                    # print('x1 shape[1]:', data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x1'].shape[1])
                    for j_h in range(data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x1'].shape[1]):
                        # hoi_id = data['hoi'][0, i_img]['id'][0, i_hoi][0, 0]
                        # action_id = metadata.hoi_to_action[hoi_id - 1]  # !NOTE: Need to subtract 1 

                        # ipdb.set_trace()
                        classname = 'worker'
                        h_x1 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x1'][0, j_h][0, 0]
                        h_y1 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['y1'][0, j_h][0, 0]
                        h_x2 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['x2'][0, j_h][0, 0]
                        h_y2 = data['hoi'][0, i_img]['bboxhuman'][0, i_hoi]['y2'][0, j_h][0, 0]
                        # print('human box:', [h_x1, h_y1, h_x2, h_y2])
                        human_index = get_node_index(classname, [h_x1, h_y1, h_x2, h_y2], det_class, det_boxes, node_num, labeled=args.labeled)
                        # print('human_index:', human_index)

                        j_o = data['hoi'][0, i_img]['connection'][0, i_hoi][j_h][1] - 1
                        # print('j_o:', j_o)
                        print('human id (j_h):', j_h, ', object id (j_o):', j_o+1)
                        print('connection:')
                        print(data['hoi'][0, i_img]['connection'][0, i_hoi])
                        print()
                        classname = list_action['nname'][hoi_id-1, 0][0]    # !NOTE: Need to subtract 1
                        o_x1 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['x1'][0, j_o][0, 0]
                        o_y1 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['y1'][0, j_o][0, 0]
                        o_x2 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['x2'][0, j_o][0, 0]
                        o_y2 = data['hoi'][0, i_img]['bboxobject'][0, i_hoi]['y2'][0, j_o][0, 0]
                        obj_index = get_node_index(classname, [o_x1, o_y1, o_x2, o_y2], det_class, det_boxes, node_num, labeled=args.labeled)
                        # print()
                        # print('gt box:', [o_x1, o_y1, o_x2, o_y2])
                        # print('det_boxes:', det_boxes)
                        # print()
                        # print('obj_index:', obj_index)
                        
                        # print("action_id:", action_id)
                        
                        if args.vis_result:
                            raw_action[action_id] = 1
                            image_gt = vis_img(image_gt, [[h_x1, h_y1, h_x2, h_y2],[o_x1, o_y1, o_x2, o_y2]], [1, metadata.coco_classes.index(classname)], raw_action=action_id, data_gt=True)

                        # print(f"human_index = {human_index} ({human_index != -1}) && obj_index = {obj_index} ({obj_index != -1})")
                        if human_index != -1 and obj_index != -1:
                            # print('labeled_edge_list:', len(labeled_edge_list))
                            edge_idx = labeled_edge_list[human_index-1] + (obj_index-human_index-1)
                            # print(f"[GET] edge_idx = {edge_idx}")
                            edge_labels[edge_idx, action_id] = 1 
                except IndexError:
                    pass
                print()
                # print()
            print("================================")
            print("edge_labels shape:", edge_labels.shape, "--> (num. possible edges givin N nodes?, num. actions)")
            print()
            print("N all nodes:", node_num)
            print("N human nodes:", human_num)
            print("num. possible edges givin N nodes:", node_num)
            print()
            # print("are all edge_labels values 0? =", (edge_labels == 0).all())
            # print()
            # visualizing result instead of saving result
            if args.vis_result:
                # ipdb.set_trace()
                image_res = Image.open(os.path.join(data_const.clean_dir, 'images/train2015', img_name)).convert('RGB')
                result = vis_img(image_res, det_boxes, det_class, det_scores, edge_labels, score_thresh=0.4)
                plt.figure(figsize=(100,100))
                plt.suptitle(img_name)
                plt.subplot(1,2,1)
                plt.imshow(np.array(image_gt))
                plt.title('all_ground_truth')
                plt.subplot(1,2,2)
                plt.imshow(np.array(result))
                plt.title('selected_ground_truth')
                # plt.axis('off')
                plt.ion()
                plt.pause(100)
                plt.close()
            # save precessed data
            else:
                save_data.create_group(global_id)
                save_data[global_id].create_dataset('node_num', data=node_num) 
                # save_data[global_id].create_dataset('img_name', data=img_name)
                save_data[global_id].create_dataset('boxes', data=det_boxes)
                save_data[global_id].create_dataset('classes', data=det_class)
                save_data[global_id].create_dataset('scores', data=det_scores)
                save_data[global_id].create_dataset('edge_labels', data=edge_labels)
                save_data[global_id].create_dataset('feature', data=feat)
        print("data.shape:", data.shape)
        print("len(save_data):", len(save_data))
        if not args.vis_result:
            save_data.close()
    # print('are all edge_labels values 0? =', (edge_labels == 0).all())
    # for el in edge_labels:
    #     print(*el)
    # create file to save images with no selected detection
    print(f"bad instance detection: <0 det>---{len(bad_dets_imgs['0'])}, <1 det>---{len(bad_dets_imgs['1'])}, <no human det>---{len(bad_dets_imgs['no_human'])}")
    io.dump_json_object(bad_dets_imgs, os.path.join('result', 'bad_faster_rcnn_det_imgs_edge.json'))

    print('Finished parsing datas!')    

def faster_rcnn_det_result(img_name):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3)
    devise = torch.device('cuda:0')
    model.cuda()
    model.eval()
    root = 'datasets/hico/images/'
    image = Image.open(os.path.join(root, 'train2015', img_name)).convert('RGB')
    input = torchvision.transforms.functional.to_tensor(image)
    input = input.to(devise)
    outputs = model([input], save_feat=False)
    # ipdb.set_trace()
    img = vis_img_frcnn(image, outputs[1][0]['boxes'].cpu().detach().numpy(), \
                         np.array(metadata.coco_pytorch_to_coco)[outputs[1][0]['labels'].cpu().detach().numpy()].astype(int), \
                         outputs[1][0]['scores'].cpu().detach().numpy(),\
                         score_thresh=0.9)
    plt.figure(figsize=(100,100))
    plt.subplot(1,1,1)
    plt.imshow(img)
    plt.axis('off')
    plt.ion()
    plt.pause(100)
    plt.close()

if __name__ == "__main__":

    parse = argparse.ArgumentParser("Parse the HICO annotion data!!!")
    parse.add_argument("--root", '-r', type=str, default='hico',
                        help="where the anno_bbox.mat file is")
    parse.add_argument("--save_dir", type=str, default='processed',
                        help="where to save the processed data")
    parse.add_argument('--vis_result', '--v_r', action="store_true", default=False,
                        help='visualize the result or not')
    parse.add_argument('--frcnn', action="store_true", default=False,
                        help='visualize the result or not')
    parse.add_argument('--labeled', action="store_true", default=False,
                        help='take instance detection label into account when getting node index')
    parse.add_argument("--feat_type", '--f_t', type=str, default='fc7', choices=['fc7', 'pool'], required=True,
                        help="which feature do you want to parse: fc7")
    parse.add_argument("--label_type", '--l_t', type=str, default='node', choices=['node', 'edge'],
                        help="which type of the label do you want: node")

    args = parse.parse_args()
    if args.frcnn:
        faster_rcnn_det_result('HICO_train2015_00014663.jpg')
    else:
        data_const = HicoConstants()
        # parse training data
        parse_data(data_const, args)