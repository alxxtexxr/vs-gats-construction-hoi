"""
Evaluation for HOI detection (Faster R-CNN detections + AGRNN action heads).

For each image in the given subset, the script:
  1. Extracts ground truth HOI detections from the dataset edge labels.
  2. Runs the AGRNN model on the Faster R-CNN detections and applies sigmoid + NMS
     (ranked by the number of confident actions per box) to obtain final detections.
  3. Matches detections against ground truth (per human/object IoU) to accumulate
     TP/FP/FN counts per action class.
  4. Reports per-class GT/TP/FP/FN and the mAP over the full action list.

Usage: python3 hico_eval.py [--checkpoint_path PATH] [--subset SUBSET]
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from datasets import metadata
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from model.model import AGRNN
from result.compute_map import compute_ap
from utils.bbox_utils import compute_iou
from utils.vis_tool import vis_img


def get_human_action_idxs(node_num, human_num):
    """Map each human index to the flattened edge indices of its action scores."""
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


def restructure_det_outputs_and_count_confident_actions(roi_labels, det_outputs, node_num, score_thresh=0.001):
    """Restructure flat det outputs into per-human action score matrices.

    Returns:
        h_det_outputs: (human_num, node_num, action_num) action scores per human.
        num_confident_actions: (node_num,) number of confident actions involving
            each node; used as the ranking score for NMS.
    """
    human_num = len(np.where(roi_labels == 1)[0])
    human_action_idxs = get_human_action_idxs(node_num, human_num)
    det_outputs_ = np.concatenate((det_outputs, np.zeros(det_outputs.shape[-1]).reshape(1, -1)), axis=0)

    h_det_outputs = np.empty((0, node_num, det_outputs.shape[-1]))
    num_confident_actions = np.zeros(node_num)

    for h_i in range(human_num):
        h_det_outputs_i = det_outputs_[human_action_idxs[h_i]]

        num_confident_actions_i = np.sum(h_det_outputs_i > score_thresh, axis=1)
        num_confident_actions += num_confident_actions_i

        h_det_outputs_i = np.expand_dims(h_det_outputs_i, axis=0)
        h_det_outputs = np.concatenate((h_det_outputs, h_det_outputs_i), axis=0)

    return h_det_outputs, num_confident_actions


def nms_by_num_confident_actions(boxes, num_confident_actions, iou_thresh=0.5):
    """NMS over detection boxes, ranked by the number of confident actions."""
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    num_confident_actions_tensor = torch.tensor(num_confident_actions, dtype=torch.float32)

    keep_idxs = torch.ops.torchvision.nms(boxes_tensor, num_confident_actions_tensor, iou_thresh)
    return keep_idxs.tolist()


def match_hoi_tp_fp(pred_det, gt_det, human_iou_thresh=0.5, object_iou_thresh=0.5):
    """Match a predicted HOI against a ground truth HOI.

    Each detection is [human_bbox(4), object_bbox(4), object_type, action_label].

    Returns:
        is_match: True if both human and object IoUs pass their thresholds and
            the action label matches.
        is_tp_human / is_tp_object: True if the corresponding IoU >= threshold.
        is_fp_human / is_fp_object: True if the corresponding IoU < threshold.
    """
    pred_human_bbox = pred_det[:4]
    pred_object_bbox = pred_det[4:8]
    pred_action = pred_det[-1]

    gt_human_bbox = gt_det[:4]
    gt_object_bbox = gt_det[4:8]
    gt_action = gt_det[-1]

    is_match = False
    is_tp_human = False
    is_tp_object = False
    is_fp_human = False
    is_fp_object = False

    human_iou = compute_iou(pred_human_bbox, gt_human_bbox)
    if human_iou >= human_iou_thresh:
        is_tp_human = True
    else:
        is_fp_human = True

    object_iou = compute_iou(pred_object_bbox, gt_object_bbox)
    if object_iou >= object_iou_thresh:
        is_tp_object = True
    else:
        is_fp_object = True

    if pred_action == gt_action:
        is_match = True

    return is_match, is_tp_human, is_tp_object, is_fp_human, is_fp_object


def compute_tp_fp_fn(pred_dets, gt_dets, num_classes, human_iou_thresh=0.5, object_iou_thresh=0.5):
    """Accumulate TP/FP/FN counts per action class.

    TP: prediction matches a ground truth IoU-wise and the action label is correct.
    FP: prediction matches a ground truth IoU-wise but the action is wrong, or no
        ground truth matches the prediction at all.
    FN: ground truth with no matching prediction.
    """
    tp_l = np.array([0] * num_classes)
    fp_l = np.array([0] * num_classes)
    fn_l = np.array([0] * num_classes)

    for gt_det in gt_dets:
        action_idx = gt_det[-1]

        found_match = False
        for pred_det in pred_dets:
            is_match, _, _, _, _ = match_hoi_tp_fp(
                pred_det, gt_det, human_iou_thresh, object_iou_thresh
            )

            if is_match:
                if pred_det[-1] == gt_det[-1]:
                    tp_l[action_idx] += 1
                else:
                    fp_l[action_idx] += 1

                found_match = True
                break

        if not found_match:
            fn_l[action_idx] += 1

    for pred_det in pred_dets:
        pred_action_idx = pred_det[-1]
        is_pred_matched = False

        for gt_det in gt_dets:
            is_match, _, _, _, _ = match_hoi_tp_fp(pred_det, gt_det, human_iou_thresh, object_iou_thresh)
            if is_match:
                is_pred_matched = True
                break

        if not is_pred_matched:
            fp_l[pred_action_idx] += 1

    return tp_l, fp_l, fn_l


def compute_pr(tp, fp, fn):
    """Compute precision and recall from scalar TP/FP/FN counts."""
    # Explicitly guard the denominator: numpy emits a RuntimeWarning (and returns
    # nan) on a 0/0 division instead of raising ZeroDivisionError, so classes
    # with no predictions/ground truth yield nan instead of noisy warnings.
    # Keep the nan a numpy scalar: a plain Python float nan breaks compute_ap's
    # scalar indexing (`precision[recall >= t]`).
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = np.float64(np.nan)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = np.float64(np.nan)
    return precision, recall


def compute_gt_counts(gt_dets, num_classes):
    """Count ground truth instances per action class."""
    gt_counts = np.array([0] * num_classes)
    for gt_det in gt_dets:
        gt_counts[gt_det[-1]] += 1
    return gt_counts


def load_model(pretrained_path, device):
    """Load a checkpointed AGRNN model onto the given device."""
    checkpoint = torch.load(pretrained_path, map_location=device)
    print('Checkpoint loaded!')

    data_const = HicoConstants(feat_type=checkpoint['feat_type'])

    model = AGRNN(
        feat_type=checkpoint['feat_type'],
        bias=checkpoint['bias'],
        bn=checkpoint['bn'],
        dropout=checkpoint['dropout'],
        layer=checkpoint['layers'],
        diff_edge=checkpoint['diff_edge'],
        multi_attn=checkpoint['multi_attn'],
        use_spatial_feat=checkpoint['use_spatial_feat'],
        use_feat=checkpoint['use_feat'],
        use_word2vec=checkpoint['use_word2vec'],
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    print('Constructed model successfully!')

    return model, data_const


def run_inference(model, data, device):
    """Run AGRNN inference on a single collated sample and apply NMS.

    Returns:
        h_det_outputs: (human_num, node_num, num_classes) per-human action scores.
        keep_idxs: indices of detection boxes kept after NMS.
    """
    node_num = data['node_num']
    features = data['features'].to(device)
    spatial_feat = data['spatial_feat'].to(device)
    word2vec = data['word2vec'].to(device)
    roi_labels = data['roi_labels']

    # roi_labels is already a list of per-image numpy arrays (batch collation),
    # matching the model's training-time signature -- do not wrap it again.
    outputs = model(node_num, features, spatial_feat, word2vec, roi_labels)

    det_outputs = nn.Sigmoid()(outputs)
    det_outputs = det_outputs.cpu().detach().numpy()

    # Post-processing helpers operate on a single image; unpack the batch-of-1.
    roi_labels_0 = roi_labels[0]

    h_det_outputs, num_confident_actions = \
        restructure_det_outputs_and_count_confident_actions(
            roi_labels_0, det_outputs, node_num[0], score_thresh=0.001)

    keep_idxs = nms_by_num_confident_actions(
        data['det_boxes'][0], num_confident_actions, iou_thresh=0.75)

    return h_det_outputs, keep_idxs


def evaluate_image(model, data, image, device, num_classes,
                   human_iou_thresh, object_iou_thresh):
    """Evaluate a single image: compute GT and prediction detection lists and
    accumulate their TP/FP/FN and GT counts."""
    # Unpack the collated batch-of-1; vis helpers expect single-image arrays.
    det_boxes = data['det_boxes'][0]
    roi_labels = data['roi_labels'][0]
    roi_labels_fasterrcnn = data['roi_labels_fasterrcnn'][0]
    edge_labels = data['edge_labels']

    # Ground truth HOI detections
    _, _, gt_h_bboxs_o_bboxs_action_class = vis_img(
        img=image,
        bboxs=det_boxes,
        labels=roi_labels,
        labels_fasterrcnn=roi_labels_fasterrcnn,
        label_type='fasterrcnn',
        raw_action=edge_labels.cpu().numpy(),
        score_thresh=0.99,
        filter_irrelevant_actions=False,
        filter_non_intersecting_det=False,
        filter_distant_det=False,
    )

    gt_counts = compute_gt_counts(gt_h_bboxs_o_bboxs_action_class, num_classes)

    # Model predictions
    h_det_outputs, keep_idxs = run_inference(model, data, device)

    _, _, det_h_bboxs_o_bboxs_action_class = vis_img(
        img=image,
        bboxs=det_boxes,
        labels=roi_labels,
        labels_fasterrcnn=roi_labels_fasterrcnn,
        keep_idxs=keep_idxs,
        label_type='fasterrcnn',
        raw_action=h_det_outputs,
        score_thresh=0.001,
        filter_irrelevant_actions=True,
        filter_non_intersecting_det=True,
        filter_distant_det=True,
    )

    tp, fp, fn = compute_tp_fp_fn(
        det_h_bboxs_o_bboxs_action_class,
        gt_h_bboxs_o_bboxs_action_class,
        num_classes,
        human_iou_thresh=human_iou_thresh,
        object_iou_thresh=object_iou_thresh,
    )

    return gt_counts, tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description='HOI Evaluation')
    parser.add_argument('--checkpoint_path', '--c_p', type=str,
                        default='checkpoints/hico/vs_gats_train/epoch_train/checkpoint_300_epoch.pth',
                        help='Path to the AGRNN checkpoint to evaluate')
    parser.add_argument('--subset', type=str, default='test',
                        choices=['train', 'test', 'train_val'],
                        help='Subset of data to run the evaluation on')
    parser.add_argument('--human_iou_thresh', type=float, default=0.5,
                        help='IoU threshold for the human bounding box match')
    parser.add_argument('--object_iou_thresh', type=float, default=0.5,
                        help='IoU threshold for the object bounding box match')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max number of images to evaluate (default: all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('Evaluating on {}...'.format(device))

    model, data_const = load_model(args.checkpoint_path, device)

    dataset = HicoDataset(data_const=data_const, subset=args.subset)
    global_ids = dataset.subset_ids
    if args.limit is not None:
        global_ids = global_ids[:args.limit]
    print(f'Evaluating on {len(global_ids)} images (subset: {args.subset})')

    num_classes = len(metadata.action_classes)
    gt_global = np.array([0] * num_classes)
    tp_l_global = np.array([0] * num_classes)
    fp_l_global = np.array([0] * num_classes)
    fn_l_global = np.array([0] * num_classes)

    for global_id in tqdm(global_ids):
        test_data = dataset.sample_date(global_id)
        if test_data is None:
            continue
        test_data = collate_fn([test_data])

        # Skip images with no processed detections
        if test_data['node_num'][0] is None:
            continue

        image_path = os.path.join(data_const.images_dir, f'{args.subset}2024', global_id + '.jpg')
        image = Image.open(image_path).convert('RGB')

        gt_counts, tp, fp, fn = evaluate_image(
            model, test_data, image, device, num_classes,
            human_iou_thresh=args.human_iou_thresh,
            object_iou_thresh=args.object_iou_thresh,
        )

        gt_global += gt_counts
        tp_l_global += tp
        fp_l_global += fp
        fn_l_global += fn

    print()
    print(f'Per-class results:')
    print("-"*64)
    print(f'{"Class":<36s} {"GT":>6s} {"TP":>6s} {"FP":>6s} {"FN":>6s}')
    print("-"*64)
    for i in range(num_classes):
        print(f'{metadata.action_classes[i]:<36s} '
              f'{gt_global[i]:>6d} {tp_l_global[i]:>6d} '
              f'{fp_l_global[i]:>6d} {fn_l_global[i]:>6d}')

    precision_l = []
    recall_l = []
    ap_l = []
    for i in range(num_classes):
        precision_i, recall_i = compute_pr(tp_l_global[i], fp_l_global[i], fn_l_global[i])
        ap_i = compute_ap(precision_i, recall_i)
        precision_l.append(precision_i)
        recall_l.append(recall_i)
        ap_l.append(ap_i)
    ap_l = np.array(ap_l)

    mAP = ap_l[~np.isnan(ap_l)].mean()

    print()
    print('Per-class precisions:')
    print("-"*64)
    print(f'{"Class":<45s} {"Precision":>18s}')
    print("-"*64)
    for idx, p in enumerate(precision_l):
        print(f'{metadata.action_classes[idx]:<45s} {p:>18g}')
    
    print()
    print(f'mAP: {mAP}')


if __name__ == '__main__':
    main()
