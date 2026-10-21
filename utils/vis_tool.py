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

# def vis_img(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8, data_gt=False):
#     try:
#         # custom
#         action_idxs = []
#         action_classes_scores = []
#         action_src_idxs = []
#         action_dst_idxs = []
        
#         if len(bboxs) == 0:
#             return img    

#         font = ImageFont.truetype(font='/root/vs-gats-2/utils/FreeMono.ttf', size=25)
#         line_width = 3
#         if data_gt:
#             img=img.convert("RGBA")
            
#             # Create a transparent overlay
#             overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
#             Drawer = ImageDraw.Draw(overlay)
#             r_color = random.choice(np.arange(256))
#             g_color = random.choice(np.arange(256))
#             b_color = random.choice(np.arange(256))
            
#             Drawer.rectangle(list(bboxs[0]), outline=(120,0,0), width=line_width)
#             Drawer.rectangle(list(bboxs[1]), outline=(120,0,0), width=line_width)
#             im_w,im_h = img.size
#             x1,y1,x2,y2 = bboxs[0]
#             x1_,y1_,x2_,y2_ = bboxs[1]
#             # import ipdb; ipdb.set_trace()
#             c_xh = int(0.5*x1)+int(0.5*x2)
#             c_yh = int(0.5*y1)+int(0.5*y2)
#             c_xo = int(0.5*x1_)+int(0.5*x2_)
#             c_yo = int(0.5*y1_)+int(0.5*y2_)
#             c_xh = max(0,min(c_xh,im_w-1))
#             c_yh = max(0,min(c_yh,im_h-1))
#             c_xo = max(0,min(c_xo,im_w-1))
#             c_yo = max(0,min(c_yo,im_h-1))

#             Drawer.line(((c_xo,c_yo),(c_xh,c_yh)), fill=(r_color,g_color,b_color,25), width=3)
            
#             text =  metadata.action_classes[raw_action]
#             # h, w = font.getsize(text)
#             _, _, h, w = font.getbbox(text)
#             Drawer.rectangle(xy=(bboxs[0][0], bboxs[0][1], bboxs[0][0]+h+1, bboxs[0][1]+w+1), fill=(r_color,g_color,b_color,25), outline=None, width=0)
#             Drawer.text(xy=(bboxs[0][0], bboxs[0][1]), text=text, font=font, fill=None)
#             return img

#         human_num = len(np.where(labels == 1)[0])
#         object_num = len(np.where(labels > 1)[0])
#         node_num = len(labels)
#         labeled_edge_list = np.cumsum(node_num - np.arange(human_num) -1)
#         labeled_edge_list[-1] = 0

#         Drawer = ImageDraw.Draw(img,"RGBA")
        
#         count = 0
#         action_idx_ = 0 # custom
#         for h_idx in range(human_num):
#             for i_idx in range(node_num):
#                 # if i_idx <= h_idx:
#                 #     continue
#                 edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
#                 action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]

#                 text = str()
#                 det_label, det_score = 'A' + str(action_idx_) + '. ', str()
#                 if len(action_idx) > 0:
#                     # custom
#                     action_src_idxs.append(h_idx)
#                     action_dst_idxs.append(i_idx)
                    
#                     r_color = random.choice(np.arange(256))
#                     g_color = random.choice(np.arange(256))
#                     b_color = random.choice(np.arange(256))

#                     text1 = 'N' + str(i_idx) + '. ' + metadata.coco_classes[labels[i_idx]]
#                     if text1 == 'pizza':
#                         continue
#                     Drawer.rectangle(list(bboxs[h_idx]), outline='#FF0000', width=line_width) # human name box
#                     Drawer.rectangle(list(bboxs[i_idx]), outline='#0000FF', width=line_width) # object name box

#                     h, w = font.getsize(text1)
#                     Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1, bboxs[i_idx][0]+h+1, bboxs[i_idx][1]), fill=(r_color,g_color,b_color,40), outline=None, width=0) # node name box
#                     Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1), text=text1, font=font, fill=None) # node name
#                     im_w,im_h = img.size
#                     x1,y1,x2,y2 = bboxs[h_idx]
#                     x1_,y1_,x2_,y2_ = bboxs[i_idx]

#                     c0 = int(0.5*x1)+int(0.5*x2)
#                     r0 = int(0.5*y1)+int(0.5*y2)
#                     c1 = int(0.5*x1_)+int(0.5*x2_)
#                     r1 = int(0.5*y1_)+int(0.5*y2_)
#                     c0 = max(0,min(c0,im_w-1))
#                     c1 = max(0,min(c1,im_w-1))
#                     r0 = max(0,min(r0,im_h-1))
#                     r1 = max(0,min(r1,im_h-1))
#                     # import ipdb; ipdb.set_trace()
#                     Drawer.line(((c0,r0),(c1,r1)), fill=(r_color,g_color,b_color), width=3)
                    
#                     # custom
#                     action_classes_scores_i = []
                    
#                     shift = 1
#                     for i in range(len(action_idx)):
#                         # custom
#                         det_label_i = metadata.action_classes[action_idx[i]]
#                         det_score_i = str(round(raw_action[edge_idx][action_idx[i]], 2))
                        
#                         det_label = det_label + " " + det_label_i
#                         det_score = det_score + "  " + det_score_i
                        
#                         # custom
#                         action_classes_scores_i.append([det_score_i, det_label_i])
                        
#                         h1, w1 = font.getsize(det_label)
#                         h2, w2 = font.getsize(det_score)
#                         # action box
#                     #     Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, 
#                     #                             bboxs[h_idx][1]-w1-w2-30, 
#                     #                             bboxs[h_idx][0]+h1+1, 
#                     #                             bboxs[h_idx][1]-30), 
#                     #                         fill=(r_color,g_color,b_color,50), 
#                     #                         outline=None, width=0)
#                     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-30), text=det_label, font=font, fill=None) # action label
#                     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-30), text=det_score, font=font, fill=None) # action score
#                     # custom
#                     action_classes_scores.append(action_classes_scores_i)
#                     action_idxs.append(action_idx_)
#                     action_idx_ += 1

#                     # # up the bbox
#                     #     Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-1), text=det_label, font=font, fill=None)
#                     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-1), text=det_score, font=font, fill=None)

#                     # # down the bbox
#                     #     Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1), text=det_label, font=font, fill=None)
#                     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_score, font=font, fill=None)

#                     # if count == 0 :
#                     #     count +=1
#                     #     # up the bbox
#                     #     for i in range(len(action_idx)):
#                     #     #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
#                     #     #     h, w = font.getsize(text)
#                     #     #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
#                     #         det_label = det_label + " " + metadata.action_classes[action_idx[i]]
#                     #         det_score = det_score + " " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
#                     #         h1, w1 = font.getsize(det_label)
#                     #         h2, w2 = font.getsize(det_score)
#                     #         # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #         Drawer.rectangle(xy=(bboxs[h_idx][0]-100, bboxs[h_idx][1]-w1-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #     Drawer.text(xy=(bboxs[h_idx][0]-100, bboxs[h_idx][1]-w1-1), text=det_label, font=font, fill=None)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-w1-1), text=det_score, font=font, fill=None)

#                     # elif count == 1:
#                     #     count+=1
#                     #     # down the bbox
#                     #     for i in range(len(action_idx)):
#                     #     #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
#                     #     #     h, w = font.getsize(text)
#                     #     #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
#                     #         det_label = det_label + " " + metadata.action_classes[action_idx[i]]
#                     #         det_score = det_score + " " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
#                     #         h1, w1 = font.getsize(det_label)
#                     #         h2, w2 = font.getsize(det_score)
#                     #         # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #         Drawer.rectangle(xy=(bboxs[h_idx][0]-0, bboxs[h_idx][3]-w1-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1), text=det_label, font=font, fill=None)
#                     #     Drawer.text(xy=(bboxs[h_idx][0]-0, bboxs[h_idx][3]-w1-1), text=det_label, font=font, fill=None)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_score, font=font, fill=None)

#                     # else:
#                     #     count+=1
#                     #     # down the bbox
#                     #     for i in range(len(action_idx)):
#                     #     #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
#                     #     #     h, w = font.getsize(text)
#                     #     #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
#                     #         det_label = det_label + " " + metadata.action_classes[action_idx[i]]
#                     #         det_score = det_score + " " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
#                     #         h1, w1 = font.getsize(det_label)
#                     #         h2, w2 = font.getsize(det_score)
#                     #         # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #         Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][3]), fill=(r_color,g_color,b_color), outline=None, width=0)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-w2-1), text=det_label, font=font, fill=None)
#                     #     Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_label, font=font, fill=None)
#                     #     # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][3]-w1-1), text=det_score, font=font, fill=None)
        
#         # custom
#         txt = ""
        
#         txt += "###### NODES ######\n"
#         node_idxs = sorted(set(action_src_idxs + action_dst_idxs))
#         node_classes = [metadata.coco_classes[labels[node_idx]] for node_idx in node_idxs]
#         for node_idx, node_class in zip(node_idxs, node_classes):
#             txt += f"{node_idx}.\t{node_class}\n"
        
#         txt += "\n"
        
#         txt += "###### ACTIONS ######\n"
#         action_data = list(zip(action_idxs, action_src_idxs, action_dst_idxs, action_classes_scores))
#         for i, (action_idx, action_src_idx, action_dst_idx, action_classes_scores_i) in enumerate(action_data):
#             txt += f"{action_idx}.\t{action_src_idx}-{action_dst_idx}\t: "
#             for j, (action_score, action_label) in enumerate(action_classes_scores_i):
#                 txt += f"{action_label} ({action_score})"
#                 if j+1 < len(action_classes_scores_i):
#                     txt += ", "
#             if i+1 < len(action_data):
#                 txt += "\n"
        
#         return img, txt

#     except Exception as e:
#         print("Error:", e)
#         print("bboxs: {}, labels: {}" .format(bboxs, labels))
#     finally:
#         pass

def vis_img(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8, data_gt=False):
    # custom
    action_idxs = []
    action_classes_scores = []
    action_src_idxs = []
    action_dst_idxs = []
    
    if len(bboxs) == 0:
        return img    

    font = ImageFont.truetype(font='/root/vs-gats-2/utils/FreeMono.ttf', size=25)
    line_width = 3
    if data_gt:
        img = img.convert('RGBA')
        # Create a transparent overlay
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        Drawer = ImageDraw.Draw(overlay)
        r_color = random.choice(np.arange(256))
        g_color = random.choice(np.arange(256))
        b_color = random.choice(np.arange(256))

        Drawer.rectangle(list(bboxs[0]), outline=(120,0,0), width=line_width)
        Drawer.rectangle(list(bboxs[1]), outline=(120,0,0), width=line_width)
        im_w,im_h = img.size
        x1,y1,x2,y2 = bboxs[0]
        x1_,y1_,x2_,y2_ = bboxs[1]
        # import ipdb; ipdb.set_trace()
        c_xh = int(0.5*x1)+int(0.5*x2)
        c_yh = int(0.5*y1)+int(0.5*y2)
        c_xo = int(0.5*x1_)+int(0.5*x2_)
        c_yo = int(0.5*y1_)+int(0.5*y2_)
        c_xh = max(0,min(c_xh,im_w-1))
        c_yh = max(0,min(c_yh,im_h-1))
        c_xo = max(0,min(c_xo,im_w-1))
        c_yo = max(0,min(c_yo,im_h-1))
        
        Drawer.line(((c_xo,c_yo),(c_xh,c_yh)), fill=(r_color,g_color,b_color, 200), width=3)
        
        text =  metadata.action_classes[raw_action]
        h, w = font.getsize(text)
        Drawer.rectangle(xy=(bboxs[0][0], bboxs[0][1], bboxs[0][0]+h+1, bboxs[0][1]+w+1), fill=(r_color,g_color,b_color, 200), outline=None, width=0)
        Drawer.text(xy=(bboxs[0][0], bboxs[0][1]), text=text, font=font, fill=None)

        return img

    human_num = len(np.where(labels == 1)[0])
    object_num = len(np.where(labels > 1)[0])
    node_num = len(labels)
    labeled_edge_list = np.cumsum(node_num - np.arange(human_num) -1)
    labeled_edge_list[-1] = 0

    Drawer = ImageDraw.Draw(img,"RGBA")
    
    # CUSTOM
    action_idx_ = 0 
    node_idxs_with_action = []
    
    for h_idx in range(human_num):
        for i_idx in range(node_num):
            if h_idx == i_idx:
                continue

            if metadata.coco_classes[labels[i_idx]] == metadata.coco_classes[labels[h_idx]] == 'worker':
                continue
            
            # if i_idx <= h_idx:
            #     continue
            
            edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
            action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]

            text = str()
            det_label, det_score = 'A' + str(action_idx_) + '. ', str()
            
            # draw bboxes for nodes that have no action
            if i_idx not in node_idxs_with_action:
                text1 = 'N' + str(i_idx) + '. ' + metadata.coco_classes[labels[i_idx]]
                i_color = '#FF0000' if metadata.coco_classes[labels[i_idx]] == 'worker' else '#0000FF'
                Drawer.rectangle(list(bboxs[i_idx]), outline=i_color, width=line_width) # object bbox
                h, w = font.getsize(text1)
                Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1, bboxs[i_idx][0]+h+1, bboxs[i_idx][1]), 
                                fill='gray', outline=None, width=0) # object label box
                Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1), text=text1, font=font, fill=None) # object label text
            
            if len(action_idx) > 0:
                # custom
                action_src_idxs.append(h_idx)
                action_dst_idxs.append(i_idx)
                node_idxs_with_action += [h_idx, i_idx]
                
                r_color = random.choice(np.arange(256))
                g_color = random.choice(np.arange(256))
                b_color = random.choice(np.arange(256))

                text1 = 'N' + str(i_idx) + '. ' + metadata.coco_classes[labels[i_idx]]
                text2 = 'N' + str(h_idx) + '. ' + metadata.coco_classes[labels[h_idx]]

                # if the label at h_idx or i_idx label is 'worker' (human), 
                # set the bbox outline color to red, else set it to blue
                h_color = '#FF0000' if metadata.coco_classes[labels[h_idx]] == 'worker' else '#0000FF'
                i_color = '#FF0000' if metadata.coco_classes[labels[i_idx]] == 'worker' else '#0000FF'
                
                Drawer.rectangle(list(bboxs[h_idx]), outline=h_color, width=line_width) # human bbox
                Drawer.rectangle(list(bboxs[i_idx]), outline=i_color, width=line_width) # object bbox

                # print(f"h_idx = {h_idx}, i_idx = {i_idx}, text2 = {text2}, text1 = {text1}")

                h, w = font.getsize(text1)
                Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1, bboxs[i_idx][0]+h+1, bboxs[i_idx][1]), 
                                    fill=i_color, outline=None, width=0) # object label box
                Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1), text=text1, font=font, fill=None) # object label text
                
                h2, w2 = font.getsize(text2)
                Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1]-w2-1, bboxs[h_idx][0]+h2+1, bboxs[h_idx][1]), 
                                    fill=h_color, outline=None, width=0) # human label box
                Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]-w2-1), text=text2, font=font, fill=None) # human label text
                
                im_w,im_h = img.size
                x1,y1,x2,y2 = bboxs[h_idx]
                x1_,y1_,x2_,y2_ = bboxs[i_idx]

                c0 = int(0.5*x1)+int(0.5*x2)
                r0 = int(0.5*y1)+int(0.5*y2)
                c1 = int(0.5*x1_)+int(0.5*x2_)
                r1 = int(0.5*y1_)+int(0.5*y2_)
                c0 = max(0,min(c0,im_w-1))
                c1 = max(0,min(c1,im_w-1))
                r0 = max(0,min(r0,im_h-1))
                r1 = max(0,min(r1,im_h-1))
                # import ipdb; ipdb.set_trace()
                Drawer.line(((c0,r0),(c1,r1)), fill=(r_color,g_color,b_color), width=5)
                
                # custom
                action_classes_scores_i = []
                
                shift = 1
                for i in range(len(action_idx)):
                    # custom
                    det_label_i = metadata.action_classes[action_idx[i]]
                    det_score_i = str(round(raw_action[edge_idx][action_idx[i]], 2))
                    
                    det_label = det_label + " " + det_label_i
                    det_score = det_score + "  " + det_score_i
                    
                    # custom
                    action_classes_scores_i.append([det_score_i, det_label_i])
                    
                    h1, w1 = font.getsize(det_label)
                    h2, w2 = font.getsize(det_score)
                    # action label box
                    Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]-1), text=det_label, font=font, fill=None)
                Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]+w1+1), text=det_score, font=font, fill=None)
                    
                action_classes_scores.append(action_classes_scores_i)
                action_idxs.append(action_idx_)
                action_idx_ += 1

    # custom
    txt = ""
    
    txt += "###### NODES ######\n"
    node_idxs = sorted(set(action_src_idxs + action_dst_idxs))
    node_classes = [metadata.coco_classes[labels[node_idx]] for node_idx in node_idxs]
    for node_idx, node_class in zip(node_idxs, node_classes):
        txt += f"{node_idx}.\t{node_class}\n"
    
    txt += "\n"
    
    txt += "###### ACTIONS ######\n"
    action_data = list(zip(action_idxs, action_src_idxs, action_dst_idxs, action_classes_scores))
    for i, (action_idx, action_src_idx, action_dst_idx, action_classes_scores_i) in enumerate(action_data):
        txt += f"{action_idx}.\t{action_src_idx}-{action_dst_idx}\t\t: "
        for j, (action_score, action_label) in enumerate(action_classes_scores_i):
            txt += f"{action_label} ({action_score})"
            if j+1 < len(action_classes_scores_i):
                txt += ", "
        if i+1 < len(action_data):
            txt += "\n"
    
    return img, txt

def vis_img_vcoco(img, bboxs, labels, scores=None, raw_action=None, score_thresh=0.8, data_gt=False):
    try:
        if len(bboxs) == 0:
            return img    

        font = ImageFont.truetype(font='/root/vs-gats-2/utils/FreeMono.ttf', size=25)
        line_width = 3
        if data_gt:
            Drawer = ImageDraw.Draw(img)
            r_color = random.choice(np.arange(256))
            g_color = random.choice(np.arange(256))
            b_color = random.choice(np.arange(256))

            Drawer.rectangle(list(bboxs[0]), outline=(120,0,0), width=line_width)
            Drawer.rectangle(list(bboxs[1]), outline=(120,0,0), width=line_width)
            im_w,im_h = img.size
            x1,y1,x2,y2 = bboxs[0]
            x1_,y1_,x2_,y2_ = bboxs[1]
            # import ipdb; ipdb.set_trace()
            c_xh = int(0.5*x1)+int(0.5*x2)
            c_yh = int(0.5*y1)+int(0.5*y2)
            c_xo = int(0.5*x1_)+int(0.5*x2_)
            c_yo = int(0.5*y1_)+int(0.5*y2_)
            c_xh = max(0,min(c_xh,im_w-1))
            c_yh = max(0,min(c_yh,im_h-1))
            c_xo = max(0,min(c_xo,im_w-1))
            c_yo = max(0,min(c_yo,im_h-1))
            
            Drawer.line(((c_xo,c_yo),(c_xh,c_yh)), fill=(r_color,g_color,b_color), width=3)
            # print(c_xo,c_yo,c_xh,c_yh)
                    
            
            text =  vcoco_metadata.action_class_with_object[raw_action]
            h, w = font.getsize(text)
            # Drawer.rectangle(xy=(bboxs[0][0], bboxs[0][1], bboxs[0][0]+h+1, bboxs[0][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
            # Drawer.text(xy=(bboxs[0][0], bboxs[0][1]), text=text, font=font, fill=None)
            Drawer.rectangle(xy=(bboxs[1][0], bboxs[1][1], bboxs[1][0]+h+1, bboxs[1][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
            Drawer.text(xy=(bboxs[1][0], bboxs[1][1]), text=text, font=font, fill=None)

            return img

        human_num = len(np.where(labels == 1)[0])
        node_num = len(labels)
        obj_num = node_num - human_num

        Drawer = ImageDraw.Draw(img)

        # count = 0
        for h_idx in range(human_num):
            same_human = 0
            for i_idx in range(node_num):
            # for i_idx in range(human_num, node_num):
                if h_idx == i_idx:
                    continue
                if h_idx == 0:
                    edge_idx = i_idx - 1
                elif h_idx > i_idx:
                    edge_idx = h_idx * (node_num-1) + i_idx
                else:
                    edge_idx = h_idx * (node_num-1) + i_idx -1
                # edge_idx = h_idx * obj_num + i_idx - human_num

                action_idx = np.where(raw_action[edge_idx] > score_thresh)[0]
                action_idx = [id for id in action_idx if id != 0]

                text = str()
                det_label, det_score = str(), str()
                if len(action_idx) > 0:
                    
                    r_color = random.choice(np.arange(256))
                    g_color = random.choice(np.arange(256))
                    b_color = random.choice(np.arange(256))

                    text1 = vcoco_metadata.coco_classes[labels[i_idx]]

                    # if text1=='laptop': continue #FF0000

                    Drawer.rectangle(list(bboxs[h_idx]+[5,0,0,0]), outline='#FF8C00', width=line_width)
                    Drawer.rectangle(list(bboxs[i_idx]), outline='#FF8C00', width=line_width)

                    h, w = font.getsize(text1)
                    # Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1], bboxs[i_idx][0]+h+1, bboxs[i_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]), text=text1, font=font, fill=None)
                    Drawer.rectangle(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1, bboxs[i_idx][0]+h+1, bboxs[i_idx][1]), fill=(r_color,g_color,b_color), outline=None, width=0)
                    Drawer.text(xy=(bboxs[i_idx][0], bboxs[i_idx][1]-w-1), text=text1, font=font, fill='#000000')
                    im_w,im_h = img.size
                    x1,y1,x2,y2 = bboxs[h_idx]
                    x1_,y1_,x2_,y2_ = bboxs[i_idx]

                    c0 = int(0.5*x1)+int(0.5*x2)
                    r0 = int(0.5*y1)+int(0.5*y2)
                    c1 = int(0.5*x1_)+int(0.5*x2_)
                    r1 = int(0.5*y1_)+int(0.5*y2_)
                    c0 = max(0,min(c0,im_w-1))
                    c1 = max(0,min(c1,im_w-1))
                    r0 = max(0,min(r0,im_h-1))
                    r1 = max(0,min(r1,im_h-1))
                    # import ipdb; ipdb.set_trace()
                    Drawer.line(((c0,r0),(c1,r1)), fill=(r_color,g_color,b_color), width=3)
                    
                    shift = 0
                    same_human += 1
                    for i in range(len(action_idx)):
                    #     text = text + " " + metadata.action_classes[action_idx[i]]+str(raw_action[edge_idx][action_idx[i]])
                    #     h, w = font.getsize(text)
                    #     Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h+1, bboxs[h_idx][1]+w+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0], bboxs[h_idx][1]), text=text, font=font, fill=None)
                        # if vcoco_metadata.action_class_with_object[action_idx[i]] != 'jump':
                        #     continue
                        det_label = det_label + vcoco_metadata.action_class_with_object[action_idx[i]]
                        # det_score = det_score + "  " + str(round(scores[h_idx] * scores[i_idx] * raw_action[edge_idx][action_idx[i]],2))
                        det_score = det_score + "  " + str(round(raw_action[edge_idx][action_idx[i]],2))
                        # import ipdb; ipdb.set_trace()
                        h1, w1 = font.getsize(det_label)
                        h2, w2 = font.getsize(det_score)
                        # Drawer.rectangle(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1
                        # ]-1, bboxs[h_idx][0]+h1+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                        Drawer.rectangle(xy=(bboxs[h_idx][0]-h1, bboxs[h_idx][2]-270, bboxs[h_idx][0]+5, bboxs[h_idx][2]-270+w1+5), fill=(r_color,g_color,b_color), outline=None, width=0)
                        # Drawer.rectangle(xy=(bboxs[h_idx][0], bboxs[h_idx][1], bboxs[h_idx][0]+h2+1, bboxs[h_idx][1]+w1+w2+1), fill=(r_color,g_color,b_color), outline=None, width=0)
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]), text=det_label, font=font, fill=None)
                    Drawer.text(xy=(bboxs[h_idx][0]-h1, bboxs[h_idx][2]-270), text=det_label, font=font, fill='#000000')
                    # Drawer.text(xy=(bboxs[h_idx][0]-shift, bboxs[h_idx][1]+w1+1), text=det_score, font=font, fill=None)
                    
        return img

    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass


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