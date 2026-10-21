import time
import ipdb

import numpy as np
import matplotlib
import torch as t
import random
# import visdom
# import cv2

matplotlib.use('Agg')
from matplotlib import pyplot as plot
from PIL import Image, ImageDraw, ImageFont
from datasets import metadata, vcoco_metadata

def is_intersecting(bbox1, bbox2):
    """
    Check if two bounding boxes intersect.

    Parameters:
    bbox1, bbox2: tuple of (x1, y1, x2, y2)
        (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    bool: True if the bounding boxes intersect, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Check if one bbox is to the left of the other
    if x1_max < x2_min or x2_max < x1_min:
        return False

    # Check if one bbox is above the other
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True

def euclidean_distance(bbox1, bbox2):
    import math
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.

    Parameters:
    bbox1, bbox2: tuple of (x1, y1, x2, y2)
        (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    float: Euclidean distance between the centers of the bounding boxes.
    """
    # Calculate the center of bbox1
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2

    # Calculate the center of bbox2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2

    # Calculate the Euclidean distance
    distance = math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)
    return distance

def calculate_2x_height(box1, box2):
    """
    Calculate 2X height from two bounding boxes and select the larger height.

    Parameters:
    - box1: list or array, bounding box [x_min, y_min, x_max, y_max] for human 1.
    - box2: list or array, bounding box [x_min, y_min, x_max, y_max] for human 2.

    Returns:
    - float: 2X of the larger height.
    """
    # Calculate heights of the two bounding boxes
    height1 = box1[3] - box1[1]  # y_max - y_min for box1
    height2 = box2[3] - box2[1]  # y_max - y_min for box2

    # Select the larger height
    larger_height = max(height1, height2)

    # Return 2X of the larger height
    return 2 * larger_height

def bbox_to_print(bbox):
    return "[" + ",".join([str(v).rjust(4) for v in np.round(bbox).astype(int).tolist()]) + "]"

def no_to_print(no):
    return (str(no) + ".").ljust(3)

def get_color(label):
    if label == 'human':
        return '#FF0000'
    elif label == 'material':
        return '#880088'
    else:
        return '#0000FF'

def vis_img(img, bboxs, labels, labels_fasterrcnn, label_type='hico', raw_action=None, score_thresh=0.8,
            filter_irrelevant_actions=True, filter_non_intersecting_det=True, filter_distant_det=True):
    if len(bboxs) == 0:
        return img    

    # font = ImageFont.truetype(font='/root/vs-gats-2/utils/FreeMono.ttf', size=25)
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf', size=25)

    line_width = 3

    human_num = len(np.where(labels == 1)[0])
    # object_num = len(np.where(labels > 1)[0])
    node_num = len(labels)
    labeled_edge_list = np.cumsum(node_num - np.arange(human_num) - 1)
    labeled_edge_list[-1] = 0

    Drawer = ImageDraw.Draw(img, "RGBA")
    
    # CUSTOM
    action_idx_ = 0 
    # node_idxs_with_action = []
    
    h_idxs = []
    i_idxs = []
    
    h_no_list = [] # Human number list
    h_classes = []
    h_bboxs = []
    
    i_no_list = [] # Object number list
    i_classes = []
    i_bboxs = []

    action_idxs = []
    action_idx =[]
    action_classes_scores = []
    action_src_idxs = []
    action_src_bboxs = []
    action_dst_idxs = []
    action_dst_bboxs = []
    action_obj_types = []

    # h_idxs_with_actions_i = 0
    # i_idxs_with_actions_i = 0
    
    for h_idx in range(human_num):
        for i_idx in range(node_num):
            if i_idx <= h_idx: 
                continue

            edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
            action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]

            # Define human and object labels
            h_label = labels[h_idx]                         # Human label (HICO)
            h_label_fasterrcnn = labels_fasterrcnn[h_idx]   # Human label (Faster R-CNN)
            i_label = labels[i_idx]                         # Object label (HICO)
            i_label_fasterrcnn = labels_fasterrcnn[i_idx]   # Object label (Faster R-CNN)
            
            # Get human and object class names
            h_class = metadata.hico_classes[h_label]                              # Human class name (HICO)
            h_class_fasterrcnn = metadata.fasterrcnn_classes[h_label_fasterrcnn]  # Human class name (Faster R-CNN)            
            i_class = metadata.hico_classes[i_label]                              # Object class name (HICO)
            i_class_fasterrcnn = metadata.fasterrcnn_classes[i_label_fasterrcnn]  # Object class name (Faster R-CNN)

            # DEBUG: Check actions
            # cond = len(action_idx) and filter_irrelevant_actions and (1 in action_idx) and ('human' in i_class)
            # if cond:
            #     print(h_class_fasterrcnn, "<-->", i_class_fasterrcnn, "=", action_idx)

            # Filter irrelevant actions
            if filter_irrelevant_actions:
                # Get relevant action classes corresponding to the object at label index `i_idx`
                relevant_actions = metadata.obj_to_hoi[i_label]

                # Filter actions that are not relevant to the object
                action_idx = [action_idx_i for action_idx_i in action_idx if action_idx_i in relevant_actions]

            text = str()
            det_label, det_score = 'A' + str(action_idx_) + '. ', str()
            
            # draw bboxes for nodes that have no action
            # if i_idx not in node_idxs_with_action:
            #     # text1 = 'N' + str(i_idx) + '. ' + metadata.hico_classes[labels[i_idx]]
            #     text1 = str(i_idx)
                
            #     i_color = '#FF0000' if metadata.hico_classes[labels[i_idx]] == 'worker' else '#0000FF'
            #     Drawer.rectangle(list(bbox_i), outline=i_color, width=line_width) # object bbox
            #     h, w = font.getsize(text1)
            #     Drawer.rectangle(xy=(bbox_i[0], bbox_i[1]-w-1, bbox_i[0]+h+1, bbox_i[1]), 
            #                     fill='gray', outline=None, width=0) # object label box
            #     Drawer.text(xy=(bbox_i[0], bbox_i[1]-w-1), text=text1, font=font, fill=None) # object label text

            # if cond:
            #     print("[FILTERED]", h_class_fasterrcnn, "<-->", i_class_fasterrcnn, "=", action_idx)
            
            # CUSTOM
            if len(action_idx) > 0:
                bbox_h = bboxs[h_idx]
                bbox_i = bboxs[i_idx]
                
                # Skip detection with non-intersecting bboxes
                if filter_non_intersecting_det and not is_intersecting(bbox_h, bbox_i):
                    continue

                # Skip detection for human-human interaction (`h_label == i_label == 1`) with distant bboxes
                if filter_distant_det and h_label == i_label == 1:
                    distance_human = euclidean_distance(bbox_h, bbox_i) 
                    distance_threshold = calculate_2x_height(bbox_h, bbox_i)

                    if distance_human > distance_threshold:
                        continue
                
                # action_src_idxs.append(h_idx)
                # action_dst_idxs.append(i_idx)
                # node_idxs_with_action += [h_idx, i_idx]

                h_idxs.append(h_idx)
                h_no = list(dict.fromkeys(h_idxs)).index(h_idx)
                if h_no not in h_no_list:
                    h_no_list.append(h_no)
                    h_classes.append(h_class_fasterrcnn if label_type == 'fasterrcnn' else h_class)
                    h_bboxs.append(bbox_h)
                action_src_idxs.append(h_no)
                action_src_bboxs.append(bbox_h)

                i_idxs.append(i_idx)
                i_no = list(dict.fromkeys(i_idxs)).index(i_idx)
                if i_no not in i_no_list:
                    i_no_list.append(i_no)
                    i_classes.append(i_class_fasterrcnn if label_type == 'fasterrcnn' else i_class)
                    i_bboxs.append(bbox_i)
                action_dst_idxs.append(i_no)
                action_dst_bboxs.append(bbox_i)
                
                r_color = random.choice(np.arange(256))
                g_color = random.choice(np.arange(256))
                b_color = random.choice(np.arange(256))

                # text_h = str(i_idx)
                # text_i = str(h_idx)

                text_h = "H" + str(h_no) +' '+ str(int(bbox_h[0]))
                text_i = "O" + str(i_no) +' '+ str(int(bbox_i[0]))

                # If the label at h_idx or i_idx label is 'worker' (human), 
                # set the bbox outline color to red, else set it to blue
                h_color = get_color(h_class)
                i_color = get_color(i_class)

                Drawer.rectangle(list(bbox_h), outline=h_color, width=line_width) # Draw the human bbox
                Drawer.rectangle(list(bbox_i), outline=i_color, width=line_width) # Draw the object bbox

                h_i, w_i = font.getsize(text_i)
                Drawer.rectangle(xy=(bbox_i[0], bbox_i[1]-w_i-1, bbox_i[0]+h_i+1, bbox_i[1]), 
                                 fill=i_color, outline=None, width=0)                           # Draw the object label box
                Drawer.text(xy=(bbox_i[0], bbox_i[1]-w_i-1), text=text_i, font=font, fill=None) # Draw the object label text
                
                h_h, w_h = font.getsize(text_h)
                Drawer.rectangle(xy=(bbox_h[0], bbox_h[1]-w_h-1, bbox_h[0]+h_h+1, bbox_h[1]), 
                                    fill=h_color, outline=None, width=0)                        # Draw the human label box
                Drawer.text(xy=(bbox_h[0], bbox_h[1]-w_h-1), text=text_h, font=font, fill=None) # Draw the human label text
                
                im_w, im_h = img.size
                x1_h, y1_h, x2_h, y2_h = bbox_h
                x1_i, y1_i, x2_i, y2_i = bbox_i

                c0 = int(0.5*x1_h) + int(0.5*x2_h)
                r0 = int(0.5*y1_h) + int(0.5*y2_h)
                c1 = int(0.5*x1_i) + int(0.5*x2_i)
                r1 = int(0.5*y1_i) + int(0.5*y2_i)
                c0 = max(0, min(c0, im_w-1))
                c1 = max(0, min(c1, im_w-1))
                r0 = max(0, min(r0, im_h-1))
                r1 = max(0, min(r1, im_h-1))
                # import ipdb; ipdb.set_trace()
                
                # Draw the action line
                Drawer.line(((c0, r0), (c1, r1)), fill=(r_color, g_color, b_color), width=2)
                
                # CUSTOM
                action_classes_scores_i = []
                
                # shift = 1
                for i in range(len(action_idx)):
                    # CUSTOM
                    det_label_idx_i = action_idx[i]
                    det_label_i = metadata.action_classes[det_label_idx_i]
                    # det_score_i = str(round(raw_action[edge_idx][action_idx[i]], 2))
                    det_score_i = raw_action[edge_idx][action_idx[i]]
                    
                    det_label = det_label + " " + det_label_i
                    det_score = det_score + "  " + str(det_score_i)
                    
                    # CUSTOM
                    action_classes_scores_i.append([det_score_i, det_label_i, det_label_idx_i])
                    
                    h1, w1 = font.getsize(det_label)
                    h2, w2 = font.getsize(det_score)

                    # Draw the action label box
                    # Drawer.rectangle(xy=(bbox_h[0]-shift, bbox_h[1]-1, bbox_h[0]+h1+1, bbox_h[1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                # Drawer.text(xy=(bbox_h[0]-shift, bbox_h[1]-1), text=det_label, font=font, fill=None)
                # Drawer.text(xy=(bbox_h[0]-shift, bbox_h[1]+w1+1), text=det_score, font=font, fill=None)
                    
                action_classes_scores.append(action_classes_scores_i)
                action_idxs.append(action_idx_)
                
                if label_type == 'fasterrcnn':
                    action_obj_types.append(labels_fasterrcnn[i_idx])
                else:
                    action_obj_types.append(labels[i_idx])
                
                action_idx_ += 1
    # CUSTOM
    txt = ""

    # node_idxs = sorted(set(action_src_idxs + action_dst_idxs))
    # node_idx_no_map = {k: v for v, k in enumerate(node_idxs)} # Mapping of node indices to sequential numbers (no.)

    # txt += "###### NODES ######\n"
    # node_classes = [metadata.hico_classes[labels[node_idx]] for node_idx in node_idxs]
    # for node_idx, node_class in zip(node_idxs, node_classes):
    #     txt += f"{node_idx}.\t{node_class}\n"

    txt += "================\nHUMANS (H)\n================\n"
    for h_no, h_class, h_bbox in zip(h_no_list, h_classes, h_bboxs):
        # txt += f"{no_to_print(h_no)} worker {bbox_to_print(h_bbox)}\n"
        txt += f"{no_to_print(h_no)} {h_class} {bbox_to_print(h_bbox)}\n"
    txt += "\n"

    txt += "================\nOBJECTS (O)\n================\n"
    for i_no, i_class, i_bbox in zip(i_no_list, i_classes, i_bboxs):
        txt += f"{no_to_print(i_no)} {i_class} {bbox_to_print(i_bbox)}\n"
    txt += "\n"
    
    txt += "================\nINTERACTIONS\n================\n"
    
    h_bboxs_o_bboxs_action_class = []

    action_data = list(zip(action_idxs, action_src_idxs, action_dst_idxs, action_classes_scores, action_src_bboxs, action_dst_bboxs, action_obj_types))
    
    for i, (action_idx, action_src_idx, action_dst_idx, action_classes_scores_i, action_src_bbox, action_dst_bbox, action_obj_type) in enumerate(action_data):
        txt += f"{no_to_print(action_idx)} H{action_src_idx} {bbox_to_print(action_src_bbox)} <--> O{action_dst_idx} {bbox_to_print(action_dst_bbox)} : "
        # Sort action classes based on their scores
        action_classes_scores_i = sorted(action_classes_scores_i, key=lambda x: x[0], reverse=True)
        
        # print(f"Check label action_obj_type {action_obj_type}")

        # Store human bounding boxes, object bounding boxes and action classes for computing mAP
        h_bboxs_o_bboxs_action_class.append((action_src_bbox.tolist() + action_dst_bbox.tolist() +[action_obj_type]+ [action_classes_scores_i[0][-1]]))
        
        for j, (action_score, action_label, action_label_idx_i) in enumerate(action_classes_scores_i):
            txt += f"{action_label} ({action_score})"
            # Add comma
            if j+1 < len(action_classes_scores_i):
                txt += ", "
        # Add line break
        if i+1 < len(action_data):
            txt += "\n"
    
    return img, txt, h_bboxs_o_bboxs_action_class

def vis_img_v2(img, bboxs, labels, labels_fasterrcnn, keep_idxs, label_type='hico', raw_action=None, score_thresh=0.8,
               filter_irrelevant_actions=True, filter_non_intersecting_det=True, filter_distant_det=True):
    if len(bboxs) == 0:
        return img    

    # font = ImageFont.truetype(font='/root/vs-gats-2/utils/FreeMono.ttf', size=25)
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf', size=25)

    line_width = 3

    human_num = len(np.where(labels == 1)[0])
    # object_num = len(np.where(labels > 1)[0])
    node_num = len(labels)
    labeled_edge_list = np.cumsum(node_num - np.arange(human_num) - 1)
    labeled_edge_list[-1] = 0

    Drawer = ImageDraw.Draw(img, "RGBA")
    
    # CUSTOM
    action_idx_ = 0 
    # node_idxs_with_action = []
    
    h_idxs = []
    i_idxs = []
    
    h_no_list = [] # Human number list
    h_classes = []
    h_bboxs = []
    
    i_no_list = [] # Object number list
    i_classes = []
    i_bboxs = []

    action_idxs = []
    action_idx =[]
    action_classes_scores = []
    action_src_idxs = []
    action_src_bboxs = []
    action_dst_idxs = []
    action_dst_bboxs = []
    action_obj_types = []

    # h_idxs_with_actions_i = 0
    # i_idxs_with_actions_i = 0

    for h_idx in range(human_num):
        if h_idx not in keep_idxs:
            continue
        for i_idx in range(node_num):
            if i_idx not in keep_idxs:
                continue
            if i_idx <= h_idx: 
                continue


            # edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
            # action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]

            action_idx = np.where(raw_action[h_idx][i_idx] > score_thresh)[0]

            # Define human and object labels
            h_label = labels[h_idx]                         # Human label (HICO)
            h_label_fasterrcnn = labels_fasterrcnn[h_idx]   # Human label (Faster R-CNN)
            i_label = labels[i_idx]                         # Object label (HICO)
            i_label_fasterrcnn = labels_fasterrcnn[i_idx]   # Object label (Faster R-CNN)
            
            # Get human and object class names
            h_class = metadata.hico_classes[h_label]                              # Human class name (HICO)
            h_class_fasterrcnn = metadata.fasterrcnn_classes[h_label_fasterrcnn]  # Human class name (Faster R-CNN)            
            h_class_fasterrcnn = (
                'human_squating' if 'human_squating' in h_class_fasterrcnn 
                else h_class_fasterrcnn
            ) # Group 'human_squating' class names
            i_class = metadata.hico_classes[i_label]                              # Object class name (HICO)
            i_class_fasterrcnn = metadata.fasterrcnn_classes[i_label_fasterrcnn]  # Object class name (Faster R-CNN)

            # Filter irrelevant actions
            if filter_irrelevant_actions:
                # Get relevant action classes
                relevant_actions = metadata.obj_to_hoi[i_label]

                # Filter actions that are not relevant to the object
                action_idx = [action_idx_i for action_idx_i in action_idx if action_idx_i in relevant_actions]

            text = str()
            det_label, det_score = 'A' + str(action_idx_) + '. ', str()
            
            # CUSTOM
            if len(action_idx) > 0:
                bbox_h = bboxs[h_idx]
                bbox_i = bboxs[i_idx]
                
                # Skip detection with non-intersecting bboxes
                if filter_non_intersecting_det and not is_intersecting(bbox_h, bbox_i):
                    continue

                # Skip detection for human-human interaction (`h_label == i_label == 1`) with distant bboxes
                if filter_distant_det and h_label == i_label == 1:
                    distance_human = euclidean_distance(bbox_h, bbox_i) 
                    distance_threshold = calculate_2x_height(bbox_h, bbox_i)

                    if distance_human > distance_threshold:
                        continue
                
                # action_src_idxs.append(h_idx)
                # action_dst_idxs.append(i_idx)
                # node_idxs_with_action += [h_idx, i_idx]

                h_idxs.append(h_idx)
                # h_no = list(dict.fromkeys(h_idxs)).index(h_idx)
                h_no = h_idx
                if h_no not in h_no_list:
                    h_no_list.append(h_no)
                    h_classes.append(h_class_fasterrcnn if label_type == 'fasterrcnn' else h_class)
                    h_bboxs.append(bbox_h)
                action_src_idxs.append(h_no)
                action_src_bboxs.append(bbox_h)

                i_idxs.append(i_idx)
                # i_no = list(dict.fromkeys(i_idxs)).index(i_idx)
                i_no = i_idx
                if i_no not in i_no_list:
                    i_no_list.append(i_no)
                    i_classes.append(i_class_fasterrcnn if label_type == 'fasterrcnn' else i_class)
                    i_bboxs.append(bbox_i)
                action_dst_idxs.append(i_no)
                action_dst_bboxs.append(bbox_i)
                
                r_color = random.choice(np.arange(256))
                g_color = random.choice(np.arange(256))
                b_color = random.choice(np.arange(256))

                # text_h = str(i_idx)
                # text_i = str(h_idx)

                text_h = "H" + str(h_no) +' '+ str(int(bbox_h[0]))
                text_i = "O" + str(i_no) +' '+ str(int(bbox_i[0]))

                # If the label at h_idx or i_idx label is 'worker' (human), 
                # set the bbox outline color to red, else set it to blue
                h_color = get_color(h_class)
                i_color = get_color(i_class)

                Drawer.rectangle(list(bbox_h), outline=h_color, width=line_width) # Draw the human bbox
                Drawer.rectangle(list(bbox_i), outline=i_color, width=line_width) # Draw the object bbox

                h_i, w_i = font.getsize(text_i)
                Drawer.rectangle(xy=(bbox_i[0], bbox_i[1]-w_i-1, bbox_i[0]+h_i+1, bbox_i[1]), 
                                 fill=i_color, outline=None, width=0)                           # Draw the object label box
                Drawer.text(xy=(bbox_i[0], bbox_i[1]-w_i-1), text=text_i, font=font, fill=None) # Draw the object label text
                
                h_h, w_h = font.getsize(text_h)
                Drawer.rectangle(xy=(bbox_h[0], bbox_h[1]-w_h-1, bbox_h[0]+h_h+1, bbox_h[1]), 
                                    fill=h_color, outline=None, width=0)                        # Draw the human label box
                Drawer.text(xy=(bbox_h[0], bbox_h[1]-w_h-1), text=text_h, font=font, fill=None) # Draw the human label text
                
                im_w, im_h = img.size
                x1_h, y1_h, x2_h, y2_h = bbox_h
                x1_i, y1_i, x2_i, y2_i = bbox_i

                c0 = int(0.5*x1_h) + int(0.5*x2_h)
                r0 = int(0.5*y1_h) + int(0.5*y2_h)
                c1 = int(0.5*x1_i) + int(0.5*x2_i)
                r1 = int(0.5*y1_i) + int(0.5*y2_i)
                c0 = max(0, min(c0, im_w-1))
                c1 = max(0, min(c1, im_w-1))
                r0 = max(0, min(r0, im_h-1))
                r1 = max(0, min(r1, im_h-1))
                # import ipdb; ipdb.set_trace()
                
                # Draw the action line
                Drawer.line(((c0, r0), (c1, r1)), fill=(r_color, g_color, b_color), width=2)
                
                # CUSTOM
                action_classes_scores_i = []
                
                # shift = 1
                for i in range(len(action_idx)):
                    # CUSTOM
                    det_label_idx_i = action_idx[i]
                    det_label_i = metadata.action_classes[det_label_idx_i]
                    # det_score_i = str(round(raw_action[edge_idx][action_idx[i]], 2))
                    # det_score_i = raw_action[edge_idx][action_idx[i]]
                    det_score_i = raw_action[h_idx][i_idx][action_idx[i]]
                    
                    det_label = det_label + " " + det_label_i
                    det_score = det_score + "  " + str(det_score_i)
                    
                    # CUSTOM
                    action_classes_scores_i.append([det_score_i, det_label_i, det_label_idx_i])
                    
                    h1, w1 = font.getsize(det_label)
                    h2, w2 = font.getsize(det_score)

                    # Draw the action label box
                    # Drawer.rectangle(xy=(bbox_h[0]-shift, bbox_h[1]-1, bbox_h[0]+h1+1, bbox_h[1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                # Drawer.text(xy=(bbox_h[0]-shift, bbox_h[1]-1), text=det_label, font=font, fill=None)
                # Drawer.text(xy=(bbox_h[0]-shift, bbox_h[1]+w1+1), text=det_score, font=font, fill=None)
                    
                action_classes_scores.append(action_classes_scores_i)
                action_idxs.append(action_idx_)
                
                if label_type == 'fasterrcnn':
                    action_obj_types.append(labels_fasterrcnn[i_idx])
                else:
                    action_obj_types.append(labels[i_idx])
                
                action_idx_ += 1
    # CUSTOM
    txt = ""

    # node_idxs = sorted(set(action_src_idxs + action_dst_idxs))
    # node_idx_no_map = {k: v for v, k in enumerate(node_idxs)} # Mapping of node indices to sequential numbers (no.)

    # txt += "###### NODES ######\n"
    # node_classes = [metadata.hico_classes[labels[node_idx]] for node_idx in node_idxs]
    # for node_idx, node_class in zip(node_idxs, node_classes):
    #     txt += f"{node_idx}.\t{node_class}\n"

    txt += "================\nHUMANS (H)\n================\n"
    for h_no, h_class, h_bbox in zip(h_no_list, h_classes, h_bboxs):
        # txt += f"{no_to_print(h_no)} worker {bbox_to_print(h_bbox)}\n"
        txt += f"{no_to_print(h_no)} {h_class} {bbox_to_print(h_bbox)}\n"
    txt += "\n"

    txt += "================\nOBJECTS (O)\n================\n"
    for i_no, i_class, i_bbox in zip(i_no_list, i_classes, i_bboxs):
        if i_no >= human_num:
            txt += f"{no_to_print(i_no)} {i_class} {bbox_to_print(i_bbox)}\n"
    txt += "\n"
    
    txt += "================\nINTERACTIONS\n================\n"
    
    h_bboxs_o_bboxs_action_class = []

    action_data = list(zip(action_idxs, action_src_idxs, action_dst_idxs, action_classes_scores, action_src_bboxs, action_dst_bboxs, action_obj_types))
    
    for i, (action_idx, action_src_idx, action_dst_idx, action_classes_scores_i, action_src_bbox, action_dst_bbox, action_obj_type) in enumerate(action_data):
        dst_type = "H" if action_dst_idx < human_num else "O"
        txt += f"{no_to_print(action_idx)} H{action_src_idx} {bbox_to_print(action_src_bbox)} <--> {dst_type}{action_dst_idx} {bbox_to_print(action_dst_bbox)} : "
        # Sort action classes based on their scores
        action_classes_scores_i = sorted(action_classes_scores_i, key=lambda x: x[0], reverse=True)
        
        # print(f"Check label action_obj_type {action_obj_type}")

        # Store human bounding boxes, object bounding boxes and action classes for computing mAP
        h_bboxs_o_bboxs_action_class.append((action_src_bbox.tolist() + action_dst_bbox.tolist() +[action_obj_type]+ [action_classes_scores_i[0][-1]]))
        
        for j, (action_score, action_label, action_label_idx_i) in enumerate(action_classes_scores_i):
            txt += f"{action_label} ({action_score})"
            # Add comma
            if j+1 < len(action_classes_scores_i):
                txt += ", "
        # Add line break
        if i+1 < len(action_data):
            txt += "\n"
    
    return img, txt, h_bboxs_o_bboxs_action_class

def vis_img_frcnn(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8):
    try:
        if len(bboxs) == 0:
            return img    

        if scores is not None:
            keep = np.where(scores > score_thresh)[0]
            bboxs = bboxs[keep]
            labels = labels[keep]
            scores = scores[keep] 
            if raw_action is not None:
                raw_action = raw_action[keep]

        line_width = 1
        color = (120,0,0)
        # build the Font object
        font = ImageFont.truetype(font='/root/vs-gats-2/utils/FreeMono.ttf', size=15)
        for idx, (bbox, label) in enumerate(zip(bboxs, labels)):
            Drawer = ImageDraw.Draw(img)
            # ipdb.set_trace()
            Drawer.rectangle(list(bbox), outline=(120,0,0), width=line_width)
            if raw_action is None:
                text = metadata.coco_classes[label]
                if scores is not None:
                    text = text + " " + '{:.3f}'.format(scores[idx])
                h, w = font.getsize(text)
                Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None, width=0)
                Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
                
            else:
                action_idx = np.where(raw_action[idx] > 0.5)[0]
                text = str()
                if len(action_idx) > 0:
                    for i in range(len(action_idx)):
                        text = text + " " + metadata.action_classes[action_idx[i]]
                        h, w = font.getsize(text)
                        Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None, width=0)
                    Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
        return img

    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass

if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    from PIL import Image, ImageDraw
    writer = SummaryWriter(log_dir='log')
    img = cv2.imread('000001.jpg')
    img2 = Image.open('demo.jpg')
    ipdb.set_trace()
    img = img.transpose(2,0,1)
    cv2.rectangle(img, (20, 20), (220,220), (0,0,255), 5)
    drawer = ImageDraw.Draw(img2)
    while True:

        writer.add_image('test', img.transpose(2,0,1)[::-1])
        writer.add_image("test2", np.array(img2).transpose(2,0,1))
        time.sleep(5)
    writer.close()