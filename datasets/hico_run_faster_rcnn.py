import os

os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')  # must be set before torch initializes the MPS allocator

import argparse
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from datasets.hico_constants import HicoConstants
from utils import io

# from utils.vis_tool import vis_img

def get_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    # NOTE: MPS is deliberately NOT part of the auto fallback. On Apple Silicon,
    # torchvision's ResNet-FPN detection models hit very slow Metal kernel paths
    # (measured ~12x slower than the same CPU), and long command buffers trip the
    # macOS GPU watchdog (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
    # Use --device=mps explicitly if you want to try it anyway.
    return "cpu"

if __name__ == "__main__":

    parse = argparse.ArgumentParser("Run Faster RCNN detection to generate instance detection data!!!")
    parse.add_argument("--faster_rcnn_path", '--f_r_p', type=str, default='faster_rcnn.pth',
                        help="path to the Faster RCNN checkpoint")
    parse.add_argument('--device', '--d', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help="which device to run the model on")
    parse.add_argument('--batch_size', '--b', type=int, default=4,
                       help="number of images per inference batch (lower this if MPS "
                            "kills long command buffers with kIOGPUCommandBufferCallbackError)")
    parse.add_argument('--num_workers', '--nw', type=int, default=4,
                       help="number of image-loading threads for prefetching")
    parse.add_argument('--half', '--fp16', action='store_true', default=False,
                       help="Run inference in half precision (FP16). Strongly recommended on "
                            "MPS: it cuts compute/memory per Metal command buffer in half, "
                            "avoiding the kIOGPUCommandBufferCallbackErrorImpactingInteractivity "
                            "watchdog timeout that aborts long FP32 forward passes.")

    args = parse.parse_args()
    
    # set up model
    device = get_device(args.device)
    checkpoint = torch.load(args.faster_rcnn_path, map_location=device)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(coco_model=False, rpn_post_nms_top_n_test=200, \
                                                                 box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3,
                                                                 num_classes=checkpoint['data']['NC'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if args.half and device in ('mps', 'cuda'):
        model = model.half()
        print('Using FP16 (half precision) inference.')
    model.eval()

    print('Beginning...')
    print(f'Device: {device}{" (auto)" if args.device == "auto" else ""}')
    print('Batch size:', args.batch_size)
    print('Number of workers:', args.num_workers)
    print('FP16:', args.half)
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    save_dir = data_const.faster_rcnn_boxes
    io.mkdir_if_not_exists(save_dir, recursive=True)

    fc7_feat_hdf5 =  os.path.join(save_dir,'faster_rcnn_fc7.hdf5')
    pool_feat_hdf5 =  os.path.join(save_dir,'faster_rcnn_pool.hdf5')
    fc7_feat = h5py.File(fc7_feat_hdf5, 'w')
    pool_feat = h5py.File(pool_feat_hdf5, 'w')

    def run_model(images):
        # Runs the model on a list of image tensors (single forward pass).
        with torch.no_grad():
            return model(images, save_feat=True)

    def load_image(index):
        # Loads and preprocesses a single image (CPU side, no model logic).
        image = Image.open(os.path.join(data_const.images_dir, anno_list[index]['image_path_postfix'])).convert('RGB')
        return torchvision.transforms.functional.to_tensor(image)

    # Prefetcher: keeps image loading/decoding running in background threads
    # so the GPU never waits on disk I/O.
    def prefetch_batches(num_batches):
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for batch in num_batches:
                futures = [executor.submit(load_image, index) for index in batch]
                yield [future.result() for future in futures]

    num_batches = [range(start, min(start + args.batch_size, len(anno_list)))
                   for start in range(0, len(anno_list), args.batch_size)]

    use_batched = args.batch_size > 1
    progress = tqdm(total=len(num_batches), desc=f'Running Faster RCNN on {len(anno_list)} images at {args.batch_size} images per batch', unit='batch')
    for batch, indices in zip(prefetch_batches(num_batches), num_batches, strict=True):
        if args.half and device in ('mps', 'cuda'):
            batch = [input.to(device).half() for input in batch]
        else:
            batch = [input.to(device) for input in batch]
        outputs = None
        if use_batched:
            try:
                outputs = run_model(batch)
                if len(outputs) != len(indices):
                    # MPS can kill a long-running command buffer mid-forward pass
                    # (kIOGPUCommandBufferCallbackErrorImpactingInteractivity), which
                    # makes the model return fewer outputs than images in the batch.
                    raise RuntimeError(
                        f'model returned {len(outputs)} outputs for a batch of {len(indices)} images')
            except RuntimeError:
                # The batched attempt is doomed on this machine (GPU watchdog or
                # memory pressure). Disable it permanently so we stop paying the
                # cost of a failed attempt on every subsequent batch.
                use_batched = False
                print('\nBatched inference failed on this device; falling back to '
                      'per-image inference for the rest of the run.')
                outputs = None
        if outputs is None:
            if device == 'mps':
                torch.mps.synchronize()
                torch.mps.empty_cache()
            # Fall back to per-image inference so one bad batch cannot abort the
            # whole run. Results are identical, only the grouping differs.
            outputs = []
            for image in batch:
                try:
                    outputs.append(run_model([image])[0])
                except RuntimeError:
                    # A single large image can still trip the watchdog; retry it
                    # once after clearing the GPU state.
                    if device == 'mps':
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    outputs.append(run_model([image])[0])
        progress.update()

        # save object detection result data
        for output, anno in zip(outputs, (anno_list[i] for i in indices), strict=True):
            # Cast back to float32 when saving so downstream consumers of these
            # features get the same dtype they would have received from FP32 inference.
            np.save(os.path.join(save_dir, '{}_boxes.npy'.format(anno['global_id'])), output['boxes'].float().cpu().detach().numpy())
            np.save(os.path.join(save_dir, '{}_scores.npy'.format(anno['global_id'])), output['scores'].float().cpu().detach().numpy())
            nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(anno['global_id']))
            io.dump_json_object(output['labels'], nms_keep_indices_path)
            fc7_feat.create_dataset(anno['global_id'], data=output['fc7_feat'].float().cpu().detach().numpy())
            pool_feat.create_dataset(anno['global_id'], data=output['pool_feat'].float().cpu().detach().numpy())
    print() 
    fc7_feat.close()
    pool_feat.close()
    print('Make detection data successfully!')
