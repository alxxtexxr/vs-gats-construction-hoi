import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io
from datasets.hico_constants import HicoConstants
from utils.bbox_utils import compute_area
from datasets.metadata_hhi import coco_classes, fasterrcnn_classes

def select_det_ids(boxes,scores,nms_keep_ids,score_thresh,max_dets, required=False):
    if nms_keep_ids is None:
        nms_keep_ids = np.arange(0,scores.shape[0])
    
    # Select non max suppressed dets
    nms_scores = scores[nms_keep_ids]
    nms_boxes = boxes[nms_keep_ids]

    # Select dets above a score_thresh and which have area > 1
    nms_ids_above_thresh = np.nonzero(nms_scores > score_thresh)[0]
    nms_ids = []
    for i in range(min(nms_ids_above_thresh.shape[0],max_dets)):
        area = compute_area(nms_boxes[i],invalid=-1)
        if area > 1:
            nms_ids.append(i)
        
    # If no dets satisfy previous criterion select the highest ranking one with area > 1
    if len(nms_ids)==0:
        if required:
            nms_ids.append(np.argmax(nms_scores))
        # for i in range(nms_keep_ids.shape[0]):
        #     area = compute_area(nms_boxes[i],invalid=-1)
        #     if area > 1:
        #         nms_ids = [i]
        #         break
        else:
            return []
    # Convert nms ids to box ids
    nms_ids = np.array(nms_ids,dtype=np.int32)
    try:
        ids = nms_keep_ids[nms_ids]
    except:
        import pdb; pdb.set_trace()

    return ids

def select_dets(
        boxes,
        scores,
        nms_keep_indices,
        exp_const):
    selected_dets = []
    
    start_end_ids = np.zeros([len(coco_classes)-1,2],dtype=np.int32)
    start_id = 0
    for cls_ind, cls_name in enumerate(coco_classes):
        if cls_ind == 0:
            # remove the predictions with background label
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        # print('cls_scores:', cls_scores)
        # print('cls_ind:', cls_ind)
        # for i, v in enumerate(nms_keep_indices):
        #     print(i, '-->', v)
        # print('len(nms_keep_indices):', len(nms_keep_indices))
        # print('len(nms_keep_indices[0]):', len(nms_keep_indices[0]))
        # print('nms_keep_indices:', nms_keep_indices)
        cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])
        # print('cls_nms_keep_ids:', cls_nms_keep_ids)
        # print()

        # guarantee at least have one person detection
        if cls_name=='human':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.human_score_thresh,
                data_const.max_num_human,
                required=True)
                
        elif cls_name=='__background__':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.background_score_thresh,
                data_const.max_num_background)
        else:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class)
        try:
            if len(select_ids)==0 :
                boxes_scores_rpn_id_label = np.empty((0,7))
        except:
            import ipdb; ipdb.set_trace()

        else:
            boxes_scores_rpn_id_label = np.concatenate((
                cls_boxes[select_ids],
                np.expand_dims(cls_scores[select_ids],1),
                np.expand_dims(select_ids,1),
                np.expand_dims([cls_ind] * len(select_ids),1)), 1)

        selected_dets.append(boxes_scores_rpn_id_label)
        num_boxes = boxes_scores_rpn_id_label.shape[0]
        start_end_ids[cls_ind-1,:] = [start_id,start_id+num_boxes]
        start_id += num_boxes

    selected_dets = np.concatenate(selected_dets)
    
    # guarantee at least have one object  
    object_selected_dets = []
    if selected_dets.shape[0] == 1:
        for cls_ind, cls_name in enumerate(coco_classes):
            if cls_ind == 0 or cls_ind == 1:
                # remove the predictions with background label
                continue
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class,
                required=True)

            if len(select_ids)==0 :
                boxes_scores_rpn_id_label = np.empty((0,7))
            else:
                boxes_scores_rpn_id_label = np.concatenate((
                    cls_boxes[select_ids],
                    np.expand_dims(cls_scores[select_ids],1),
                    np.expand_dims(select_ids,1),
                    np.expand_dims([cls_ind] * len(select_ids),1)), 1)

            object_selected_dets.append(boxes_scores_rpn_id_label)

        object_selected_dets = np.concatenate(object_selected_dets)
        max_score_idx = np.argmax(object_selected_dets[:,4])
        object_selected_det = object_selected_dets[max_score_idx,:]
        
        try:
            selected_dets = np.concatenate((selected_dets, object_selected_det[None, :]))
            start_end_ids[int(object_selected_det[6]-1)] = [1,2]
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
        
        
    return selected_dets, start_end_ids

# CUSTOM
# def select_dets_custom(
#     boxes,
#     scores,
#     nms_keep_indices,
#     exp_const,
#     worker_idx=1,
#     supervisor_idx=8,
# ):
#     selected_dets = []

#     start_end_ids = np.zeros([len(coco_classes)-1,2],dtype=np.int32)
#     start_id = 0
    
#     for cls_ind, cls_name in enumerate(coco_classes):
#         if cls_ind == 0:
#             # remove the predictions with background label
#             continue
        
#         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

#         # guarantee at least have one person detection
#         if cls_name == 'worker':
#             select_ids = select_det_ids(
#                 cls_boxes,
#                 cls_scores,
#                 cls_nms_keep_ids,
#                 data_const.human_score_thresh,
#                 data_const.max_num_human,
#                 required=True)
                
#         elif cls_name == '__background__':
#             select_ids = select_det_ids(
#                 cls_boxes,
#                 cls_scores,
#                 cls_nms_keep_ids,
#                 data_const.background_score_thresh,
#                 data_const.max_num_background)
#         else:
#             select_ids = select_det_ids(
#                 cls_boxes,
#                 cls_scores,
#                 cls_nms_keep_ids,
#                 data_const.object_score_thresh,
#                 data_const.max_num_objects_per_class)
#         try:
#             if len(select_ids)==0 :
#                 boxes_scores_rpn_id_label = np.empty((0,7))
#         except:
#             import ipdb; ipdb.set_trace()

#         else:
#             boxes_scores_rpn_id_label = np.concatenate((
#                 cls_boxes[select_ids],
#                 np.expand_dims(cls_scores[select_ids],1),
#                 np.expand_dims(select_ids,1),
#                 np.expand_dims([cls_ind] * len(select_ids),1)), 1)

#         # Extra 'boxes_scores_rpn_id_label' for cls_ind = worker_idx (1) from cls_ind_x = supervisor_idx (8)
#         if cls_ind == worker_idx:
#             cls_ind_x = supervisor_idx
#             cls_boxes_x = boxes[:, 4*cls_ind_x:4*(cls_ind_x + 1)]
#             cls_scores_x = scores[:, cls_ind_x]
#             cls_nms_keep_ids_x = np.array(nms_keep_indices[cls_ind_x])
            
#             select_ids_x = select_det_ids(
#                 cls_boxes_x,
#                 cls_scores_x,
#                 cls_nms_keep_ids_x,
#                 data_const.human_score_thresh,
#                 data_const.max_num_human,
#                 required=True)
            
#             try:
#                 if len(select_ids_x) == 0 :
#                     boxes_scores_rpn_id_label_x = np.empty((0, 7))
#             except:
#                 import ipdb; ipdb.set_trace()

#             else:
#                 boxes_scores_rpn_id_label_x = np.concatenate((
#                     cls_boxes[select_ids_x],
#                     np.expand_dims(cls_scores_x[select_ids_x], axis=1),
#                     np.expand_dims(select_ids_x, axis=1),
#                     np.expand_dims([cls_ind] * len(select_ids_x), 1)), axis=1)
                
#                 print("boxes_scores_rpn_id_label shape:", boxes_scores_rpn_id_label.shape)
#                 print("boxes_scores_rpn_id_label_x (x for extra) shape:", boxes_scores_rpn_id_label_x.shape)
#                 boxes_scores_rpn_id_label = np.concatenate((boxes_scores_rpn_id_label, boxes_scores_rpn_id_label_x), axis=0)
#                 print("boxes_scores_rpn_id_label shape (with extra):", boxes_scores_rpn_id_label.shape)
#                 print()

#         selected_dets.append(boxes_scores_rpn_id_label)
#         num_boxes = boxes_scores_rpn_id_label.shape[0]
#         start_end_ids[cls_ind-1, :] = [start_id,start_id+num_boxes]
#         start_id += num_boxes
    
#     selected_dets = np.concatenate(selected_dets)
    
#     # guarantee at least have one object  
#     object_selected_dets = []
#     if selected_dets.shape[0] == 1:
#         for cls_ind, cls_name in enumerate(coco_classes):
#             if cls_ind == 0 or cls_ind == 1:
#                 # remove the predictions with background label
#                 continue
#             cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#             cls_scores = scores[:, cls_ind]
#             cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

#             select_ids = select_det_ids(
#                 cls_boxes,
#                 cls_scores,
#                 cls_nms_keep_ids,
#                 data_const.object_score_thresh,
#                 data_const.max_num_objects_per_class,
#                 required=True)

#             if len(select_ids)==0 :
#                 boxes_scores_rpn_id_label = np.empty((0,7))
#             else:
#                 boxes_scores_rpn_id_label = np.concatenate((
#                     cls_boxes[select_ids],
#                     np.expand_dims(cls_scores[select_ids],1),
#                     np.expand_dims(select_ids,1),
#                     np.expand_dims([cls_ind] * len(select_ids),1)), 1)

#             object_selected_dets.append(boxes_scores_rpn_id_label)

#         object_selected_dets = np.concatenate(object_selected_dets)
#         max_score_idx = np.argmax(object_selected_dets[:,4])
#         object_selected_det = object_selected_dets[max_score_idx,:]
        
#         try:
#             selected_dets = np.concatenate((selected_dets, object_selected_det[None, :]))
#             start_end_ids[int(object_selected_det[6]-1)] = [1,2]
#         except Exception as e:
#             print(e)
#             import ipdb; ipdb.set_trace()

#     return selected_dets, start_end_ids

def get_idxs_by_prefix(prefix, lst):
    return [idx for idx, v in enumerate(lst) if v.startswith(prefix + '_')]

# CUSTOM
# def select_dets_custom(
#     boxes,
#     scores,
#     nms_keep_indices,
# ):
#     selected_dets = []

#     start_end_ids = np.zeros([len(coco_classes)-1, 2],dtype=np.int32)
#     start_id = 0
    
#     for cls_ind, cls_name in enumerate(coco_classes):
#         if cls_ind == 0:
#             # Remove the predictions with background label
#             continue
        
#         # Example:
#         # fasterrcnn_classes = ['object_x', 'human_x', 'human_y']
#         # get_idxs_by_prefix('human', fasterrcnn_classes) # Output: [1, 2]
#         # get_idxs_by_prefix('object', fasterrcnn_classes) # Output: [0]
#         cls_x_idxs = get_idxs_by_prefix(cls_name, fasterrcnn_classes)
#         # print("cls_name:", cls_name)
#         # print("cls_x_idxs:", cls_x_idxs)
#         # print()

#         boxes_scores_rpn_id_label_list = []

#         for cls_x_idx in cls_x_idxs:
#             cls_boxes = boxes[:, 4*cls_x_idx:4*(cls_x_idx + 1)]
#             cls_scores = scores[:, cls_x_idx]
#             cls_nms_keep_ids = np.array(nms_keep_indices[cls_x_idx])

#             if cls_name == 'human':
#                 select_ids = select_det_ids(
#                     cls_boxes,
#                     cls_scores,
#                     cls_nms_keep_ids,
#                     data_const.human_score_thresh,
#                     data_const.max_num_human,
#                     required=True)
#             # elif cls_name == '__background__':
#             #     select_ids = select_det_ids(
#             #         cls_boxes,
#             #         cls_scores,
#             #         cls_nms_keep_ids,
#             #         data_const.background_score_thresh,
#             #         data_const.max_num_background)
#             else:
#                 select_ids = select_det_ids(
#                     cls_boxes,
#                     cls_scores,
#                     cls_nms_keep_ids,
#                     data_const.object_score_thresh,
#                     data_const.max_num_objects_per_class)
            
#             try:
#                 if len(select_ids) == 0:
#                     boxes_scores_rpn_id_label = np.empty((0, 7))
#             except:
#                 import ipdb; ipdb.set_trace()
#             else:
#                 boxes_scores_rpn_id_label = np.concatenate((
#                     cls_boxes[select_ids],
#                     np.expand_dims(cls_scores[select_ids], axis=1),
#                     np.expand_dims(select_ids, axis=1),
#                     np.expand_dims([cls_ind] * len(select_ids), axis=1),
#                 ), axis=1)
#                 # print("cls_x_name:", fasterrcnn_classes[cls_x_idx])
#                 # print("boxes_scores_rpn_id_label.shape:", boxes_scores_rpn_id_label.shape)
#                 # print(boxes_scores_rpn_id_label.shape, end=", ")

#             boxes_scores_rpn_id_label_list.append(boxes_scores_rpn_id_label)
        
#         boxes_scores_rpn_id_label = np.concatenate(boxes_scores_rpn_id_label_list, axis=0)
#         selected_dets.append(boxes_scores_rpn_id_label)
#         num_boxes = boxes_scores_rpn_id_label.shape[0]
#         start_end_ids[cls_ind-1, :] = [start_id,start_id+num_boxes]
#         start_id += num_boxes
    
#     selected_dets = np.concatenate(selected_dets)
    
#     # Guarantee at least have one object  
#     # object_selected_dets = []
#     # if selected_dets.shape[0] == 1:
#     #     for cls_ind, cls_name in enumerate(coco_classes):
#     #         if cls_ind == 0 or cls_ind == 1:
#     #             # Remove the predictions with background label
#     #             continue
#     #         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#     #         cls_scores = scores[:, cls_ind]
#     #         cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

#     #         select_ids = select_det_ids(
#     #             cls_boxes,
#     #             cls_scores,
#     #             cls_nms_keep_ids,
#     #             data_const.object_score_thresh,
#     #             data_const.max_num_objects_per_class,
#     #             required=True)

#     #         if len(select_ids)==0 :
#     #             boxes_scores_rpn_id_label = np.empty((0,7))
#     #         else:
#     #             boxes_scores_rpn_id_label = np.concatenate((
#     #                 cls_boxes[select_ids],
#     #                 np.expand_dims(cls_scores[select_ids],1),
#     #                 np.expand_dims(select_ids,1),
#     #                 np.expand_dims([cls_ind] * len(select_ids),1)), 1)

#     #         object_selected_dets.append(boxes_scores_rpn_id_label)

#     #     object_selected_dets = np.concatenate(object_selected_dets)
#     #     max_score_idx = np.argmax(object_selected_dets[:,4])
#     #     object_selected_det = object_selected_dets[max_score_idx,:]
        
#     #     try:
#     #         selected_dets = np.concatenate((selected_dets, object_selected_det[None, :]))
#     #         start_end_ids[int(object_selected_det[6]-1)] = [1,2]
#     #     except Exception as e:
#     #         print(e)
#     #         import ipdb; ipdb.set_trace()

#     return selected_dets, start_end_ids

# CUSTOM
def select_dets_custom(
    boxes,
    scores,
    nms_keep_indices,
):
    selected_dets = []
    
    start_end_ids = np.zeros([len(fasterrcnn_classes)-1, 2], dtype=np.int32)
    start_id = 0
    for cls_ind, cls_name in enumerate(fasterrcnn_classes):
        if cls_ind == 0:
            # Remove the predictions with background label
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        # print('cls_scores:', cls_scores)
        # print('cls_ind:', cls_ind)
        # for i, v in enumerate(nms_keep_indices):
        #     print(i, '-->', v)
        # print('len(nms_keep_indices):', len(nms_keep_indices))
        # print('len(nms_keep_indices[0]):', len(nms_keep_indices[0]))
        # print('nms_keep_indices:', nms_keep_indices)
        cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])
        # print('cls_nms_keep_ids:', cls_nms_keep_ids)
        # print()

        # guarantee at least have one person detection
        if 'worker' in cls_name or 'human' in cls_name:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.human_score_thresh,
                data_const.max_num_human,
                required=True)
                
        elif cls_name == '__background__':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.background_score_thresh,
                data_const.max_num_background)
        else:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class)
        try:
            if len(select_ids)==0 :
                boxes_scores_rpn_id_label = np.empty((0,7))
        except:
            import ipdb; ipdb.set_trace()

        else:
            boxes_scores_rpn_id_label = np.concatenate((
                cls_boxes[select_ids],
                np.expand_dims(cls_scores[select_ids],1),
                np.expand_dims(select_ids,1),
                np.expand_dims([cls_ind] * len(select_ids),1)), 1)

        selected_dets.append(boxes_scores_rpn_id_label)
        num_boxes = boxes_scores_rpn_id_label.shape[0]
        start_end_ids[cls_ind-1,:] = [start_id,start_id+num_boxes]
        start_id += num_boxes

    selected_dets = np.concatenate(selected_dets)
    
    # guarantee at least have one object  
    object_selected_dets = []
    if selected_dets.shape[0] == 1:
        for cls_ind, cls_name in enumerate(coco_classes):
            if cls_ind == 0 or cls_ind == 1:
                # remove the predictions with background label
                continue
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                data_const.object_score_thresh,
                data_const.max_num_objects_per_class,
                required=True)

            if len(select_ids)==0 :
                boxes_scores_rpn_id_label = np.empty((0,7))
            else:
                boxes_scores_rpn_id_label = np.concatenate((
                    cls_boxes[select_ids],
                    np.expand_dims(cls_scores[select_ids],1),
                    np.expand_dims(select_ids,1),
                    np.expand_dims([cls_ind] * len(select_ids),1)), 1)

            object_selected_dets.append(boxes_scores_rpn_id_label)

        object_selected_dets = np.concatenate(object_selected_dets)
        max_score_idx = np.argmax(object_selected_dets[:,4])
        object_selected_det = object_selected_dets[max_score_idx,:]
        
        try:
            selected_dets = np.concatenate((selected_dets, object_selected_det[None, :]))
            start_end_ids[int(object_selected_det[6]-1)] = [1,2]
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
        
    return selected_dets, start_end_ids

def select(data_const):
    io.mkdir_if_not_exists(data_const.proc_dir)
    
    select_boxes_dir = data_const.proc_dir

    # Print where the boxes are coming from and where the output is written
    print(f'Boxes will be read from: {data_const.faster_rcnn_boxes}')
    print(f'Boxes will be written to: {select_boxes_dir}')

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)

    print('Creating selected_coco_cls_dets.hdf5 file ...')
    # hdf5_file = os.path.join(select_boxes_dir,'selected_coco_cls_dets_0.1eval.hdf5')
    hdf5_file = os.path.join(select_boxes_dir,'selected_coco_cls_dets.hdf5')
    f = h5py.File(hdf5_file,'w')

    print('Selecting boxes ...')
    for anno in tqdm(anno_list):
        global_id = anno['global_id']

        # # get more detection for evaluation
        # if 'test' in global_id:
        #     data_const.human_score_thresh = 0.1
        #     data_const.object_score_thresh = 0.1

        boxes_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_boxes.npy')
        
        if not os.path.exists(boxes_npy):
            print("Skipped because boxes_npy does not exist:", boxes_npy)
            continue

        # print('boxes_npy:', boxes_npy)
        boxes = np.load(boxes_npy)
        
        scores_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_scores.npy')
        scores = np.load(scores_npy)
        
        nms_keep_indices_json = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_nms_keep_indices.json')
        nms_keep_indices = io.load_json_object(nms_keep_indices_json)

        # import ipdb; ipdb.set_trace()
        # selected_dets, start_end_ids = select_dets(boxes,scores,nms_keep_indices,data_const)
        selected_dets, start_end_ids = select_dets_custom(boxes, scores, nms_keep_indices) #, data_const, worker_idx=1, supervisor_idx=8)

        # Convert Label Faster RCNN from 161 class become 
        # DEBUG
        # print("selected_dets (worker_idx --> supervisor_idx):")
        # print(selected_dets[:, -1])
        
        f.create_group(global_id)
        f[global_id].create_dataset('boxes_scores_rpn_ids',data=selected_dets)
        f[global_id].create_dataset('start_end_ids',data=start_end_ids)
    f.close()

if __name__ == "__main__":
    data_const = HicoConstants()
    select(data_const)