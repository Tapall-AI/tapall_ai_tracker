from hydra import compose, initialize
import os
from tracker import BaseTracker
import glob
from PIL import Image, ExifTags
import numpy as np
import torch
from tqdm import tqdm
import cv2
from torch.functional import F
import torch.multiprocessing as mp

'''
python cutie/eval_vos.py dataset=mose weights=/ssd1/yk/Cutie/ckpt/r101_resume_large_4_512_main_training_last.pth \
model=large dataset=mose-val max_mem_frames=15 mem_every=3 datasets.mose-val.size=752 \
datasets.mose-val.mem_every=3 exp_id=mose_large_maxmem=15
'''


initialize(version_base='1.3.2', config_path='config', job_name="multiscale")
crt_dir = os.path.dirname(os.path.abspath(__file__))

def init_one_tracker(gpu = 'cuda:4') -> BaseTracker:
    return BaseTracker(gpu, compose(config_name=f"gui_config"))

# init multiple trackers
def init_trackers(init_image = None, init_mask = None, multi_res = [600, 720, 800], gpus = [0, 1, 2], cfg_name = 'gui_config') -> BaseTracker:
    trackers = []
    cnt = 0
    for res in multi_res:
        cfg = compose(config_name=cfg_name)
        trackers.append(BaseTracker(f'cuda:{gpus[cnt]}', cfg))
        trackers[-1].tracker.max_internal_size = res
        cnt+=1
    if init_image is not None and init_mask is not None:
        for tracker in trackers:
            tracker.track(init_image, init_mask)
    return trackers

def vots_track(trackers, image, num_objects):
    prob = torch.zeros((len(trackers), num_objects + 1, image.shape[0], image.shape[1]), dtype=torch.float32, device='cpu')
    for i, tracker in enumerate(trackers):
        _, tmp, _ = tracker.track(image)
        prob[i] = tmp.to('cpu')

    prob = torch.mean(prob, axis=0)
    final_mask = trackers[0].tracker.output_prob_to_mask(prob).cpu().numpy()        
    return final_mask


def generate_masks(frame_path, ref_path, mask_path, ref_frames, gpu='cuda:3', painted_path = ''):
    '''
    Input:
        frame_path: path to the image folder
        ref_path: path to the reference mask folder
        mask_path: path to the output mask folder
        ref_frames: list of reference frame numbers
        gpu: gpu number
    Output:
        save masks to mask_path
    '''
    # config & initialization
    device = gpu
    initialize(version_base='1.3.2', config_path='config', job_name="gui")
    cfg = compose(config_name="gui_config")
    tracker = BaseTracker(device, cfg)
    
    # load in permanent frames & masks
    image_path_list = glob.glob(os.path.join(frame_path, '*.png'))
    image_path_list.sort()
    mask_path_list = glob.glob(os.path.join(ref_path, '*.png'))
    mask_path_list.sort()

    # load frames and masks
    frames = []
    permanent_masks = []
    permanent_frames = []
    mask_id = 0
    for i, img_path in enumerate(image_path_list):
        # img = correct_image_orientation(img_path)
        crt_img = np.array(Image.open(img_path).convert('RGB'))
        frames.append(crt_img)
        if i+1 in ref_frames:
            permanent_frames.append(crt_img)
            # permanent_masks.append(np.array(Image.open(mask_path_list[mask_id]).convert('P')) //255)
            image = Image.open(mask_path_list[mask_id]).convert('RGBA')
            alpha_channel = image.split()[-1]
            alpha_array = np.array(alpha_channel)
            alpha_array[alpha_array > 0] = 1
            permanent_masks.append(alpha_array)
            mask_id += 1
    frames = np.stack(frames, 0)    # T, H, W, C

    permanent_outputs = tracker.load_permanent_memory(permanent_frames, permanent_masks)

    painted_frames = []
    out_masks = []
    mi = 0
    for ti, frame in tqdm(enumerate(frames)):
        if ti+1 in ref_frames:
            mask, prob, painted_frame = tracker.track(permanent_frames[mi], permanent_masks[mi])
            mi += 1
        else:
            mask, prob, painted_frame = tracker.track(frame)
        painted_frames.append(painted_frame)
        out_masks.append(mask)

    tracker.clear_memory()
    print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path,exist_ok=True)
    ti = 0
    for mask in tqdm(out_masks):
        mask = (mask > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        mask.save(os.path.join(mask_path, f"frame_{ti+1:05d}.png"))
        ti += 1
    if painted_path:
        if not os.path.exists(painted_path):
            os.makedirs(painted_path,exist_ok=True)
        for i, painted_frame in enumerate(tqdm(painted_frames)):
            painted_frame = Image.fromarray(painted_frame)
            painted_frame.save(os.path.join(painted_path, f"frame_{i+1:05d}.png"))

if __name__ == '__main__':
    base_path = '/home/yk/tapall_projects/threeD_Com_Insert/tmp/05_23_18_03_01_bd9a'
    frame_path = base_path + '/images'
    ref_path = base_path + '/ref_mask'
    mask_path = base_path + '/ws/mask'
    ref_frames = [1]
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    generate_masks(frame_path, ref_path, mask_path, ref_frames, gpu='cuda:1', painted_path = base_path + '/ws/painted')