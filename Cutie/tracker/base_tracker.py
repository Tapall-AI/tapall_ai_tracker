"""
Dense Tracking from Multi-Frame Annotations
This project is made by Tapall.AI, based on 2 awesome works: Cutie and XMem++
https://github.com/hkchengrex/Cutie
https://github.com/max810/XMem2
"""
import os
import glob
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from omegaconf import DictConfig
from hydra import compose, initialize
from torchvision.transforms.functional import to_tensor


from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from tracker.tools.painter import mask_painter

class BaseTracker:
    def __init__(self, device, cfg: DictConfig) -> None:
        """
        device: model device
        cfg: configurations
        """
        self.device = device
        self.cfg = cfg
        self.cutie = CUTIE(self.cfg).eval().to(self.device)
        model_weights = torch.load(self.cfg.weights, map_location=self.device)
        self.cutie.load_weights(model_weights)
        self.tracker = InferenceCore(self.cutie, self.cfg)
        print('BaseTracker Loaded.')

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def track(self, frame, annotation=None, idx_mask=True):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """
        if annotation is not None:   # first frame mask
            # initialisation
            mask_tensor = torch.Tensor(annotation).to(self.device)
            objects = np.unique(np.array(annotation))
            objects = objects[objects != 0].tolist()  # background "0" does not count as an object
        else:
            mask_tensor = None
            objects = None
        # prepare inputs
        frame_tensor = to_tensor(frame).to(self.device)
        # track one frame
        output_prob = self.tracker.step(frame_tensor, mask_tensor, objects=objects, idx_mask=idx_mask)   # logits 2 (bg fg) H W
        # convert output probabilities to an object mask
        final_mask = self.tracker.output_prob_to_mask(output_prob).cpu().numpy()

        # draw masks
        painted_image = frame
        # objects = np.unique(final_mask)
        # objects = objects[objects != 0].tolist()  # background "0" does not count as an object
        
        # for obj in objects:
        #     if np.max(final_mask==obj) == 0:
        #         continue
        #     painted_image = mask_painter(painted_image, (final_mask==obj).astype('uint8'), mask_color=obj+1)
        # print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')
        return final_mask, output_prob, painted_image

    @torch.no_grad()
    def load_permanent_memory(self, p_frames, p_masks):
        """
        load permanent memory into XMem
        p_frames: List[numpy arrays: (H, W, 3)]
        p_masks: List[numpy arrays: (H, W)]
        """
        output_probs = []
        for (frame, mask) in zip(p_frames, p_masks):
            objects = np.unique(np.array(mask))
            objects = objects[objects != 0].tolist()                    # background "0" does not count as an object
            frame_tensor = to_tensor(frame).to(self.device)             # 3, H, W
            mask_tensor = torch.Tensor(mask).to(self.device)            # n_objs, H, W
            output_probs.append(self.tracker.step(frame_tensor, mask_tensor, objects=objects, force_permanent=True))
        return output_probs

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        with torch.cuda.device(int(self.device[-1])):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # video frames (take videos from DAVIS-2017 as examples)
    video_path_list = glob.glob(os.path.join('/home/yk/tapall_projects/threeD_Com_Insert/tmp/05_22_10_56_19_7e64/images', '*.png'))
    video_path_list.sort()
    mask_path_list = glob.glob(os.path.join('/home/yk/tapall_projects/threeD_Com_Insert/tmp/05_22_10_56_19_7e64/ref_mask', '*.png'))
    mask_path_list.sort()

    # mask indicator
    mask_ids = [0,6,18,29]

    # load frames and masks
    frames = []
    permanent_masks = []
    permanent_frames = []
    mask_id = 0
    for i, video_path in enumerate(video_path_list):
        frames.append(np.array(Image.open(video_path).convert('RGB')))
        if i in mask_ids:
            permanent_frames.append(np.array(Image.open(video_path).convert('RGB')))
            permanent_masks.append(np.array(Image.open(mask_path_list[mask_id]).convert('P'))//255)
            mask_id += 1
    frames = np.stack(frames, 0)    # T, H, W, C

    # ------------------------------------------------------------------------------------
    # how to use
    # ------------------------------------------------------------------------------------
    # 1/5: set checkpoint and device
    device = 'cuda:0'
    initialize(version_base='1.3.2', config_path="../cutie/config", job_name="gui")
    cfg = compose(config_name="gui_config")
    # ------------------------------------------------------------------------------------
    # 2/5: initialise tracker
    tracker = BaseTracker(device, cfg)
    # ------------------------------------------------------------------------------------
    # 3/5: initialise permanent memory
    permanent_outputs = tracker.load_permanent_memory(permanent_frames, permanent_masks)
    # ------------------------------------------------------------------------------------
    # 4/5: for each frame, get tracking results by tracker.track(frame, first_frame_annotation)
    # frame: numpy array (H, W, C), first_frame_annotation: numpy array (H, W), leave it blank when tracking begins
    painted_frames = []
    mi = 0
    for ti, frame in enumerate(frames):
        if ti in mask_ids:
            mask, prob, painted_frame = tracker.track(permanent_frames[mi], permanent_masks[mi])
            mi += 1
        else:
            mask, prob, painted_frame = tracker.track(frame)
        painted_frames.append(painted_frame)
    # ----------------------------------------------
    # 3/4: clear memory in XMEM for the next video
    tracker.clear_memory()
    # ----------------------------------------------
    # end
    # ----------------------------------------------
    print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')
    # set saving path
    save_path = '/home/yk/tapall_projects/threeD_Com_Insert/tmp/05_22_10_56_19_7e64/ws/mask'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # save
    ti = 0
    for painted_frame in tqdm(painted_frames):
        painted_frame = Image.fromarray(painted_frame)
        painted_frame.save(f'{save_path}/{ti:05d}.png')
        ti += 1
