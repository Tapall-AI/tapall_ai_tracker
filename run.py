import vot
import cv2
from cutie.segmentation import init_trackers, vots_track
from cutie import mask_painter
import numpy as np




def make_full_size(mask, shape):
    full_size_mask = np.zeros(shape, dtype=np.uint8)
    full_size_mask[:mask.shape[0], :mask.shape[1]] = mask
    return full_size_mask

def init_segmentation(config_file, model_file):
    register_all_modules()
    model = init_detector(config_file, model_file, device='cuda:1')
    return model

def run_vots2024_exp(multi_res = [720, 800]):
    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()
    imagefile = handle.frame()
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    # 一般以h,w的形式存储 其中h为短边
    ori_shape = image.shape[:2] 
    # 预处理将多个mask合并为一个mask
    init_mask = np.zeros_like(image[:, :, 0])
    num_objects = len(objects)  
    for idx, obj in enumerate(objects):
        init_mask += make_full_size(obj, ori_shape) * (idx + 1)
    

    trackers = init_trackers(init_image=image, init_mask=init_mask, multi_res=multi_res, gpus=[3,4])
    # 计数器（调试用）
    cnt = 0

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        mask = vots_track(trackers, image, num_objects = num_objects)
        # 分离mask中不同id的mask
        objs = np.unique(mask)
        objs = objs[objs != 0].tolist()  # background "0" does not count as an object
        cnt += 1
        all_objs = [(mask == i + 1).astype('uint8') for i in range(num_objects)]
        handle.report(all_objs) 

if __name__ == '__main__':
    run_vots2024_exp()



