from os import path
import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize

import torch
from torch.utils.data import DataLoader

from cutie.inference.data.vos_test_dataset import VOSTestDataset
from cutie.inference.data.burst_test_dataset import BURSTTestDataset
from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.results_utils import ResultSaver, make_zip
from cutie.inference.utils.burst_utils import BURSTResultHandler
from cutie.inference.utils.args_utils import get_dataset_cfg
# from masa.vots_entrypoint import InstanceSegmenter
import numpy as np
from tracker import BaseTracker
from tqdm import tqdm
import redis
import torch.multiprocessing as mp
import pickle
log = logging.getLogger()
from hydra.core.global_hydra import GlobalHydra


REDIS_HOST = 'localhost'
REDIS_PORT = 6379
FRAME_CHANNEL = 'frame_channel222'
RESULT_CHANNEL = 'result_channel111'

pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT)



def init_one_tracker(gpu = 'cuda:4') -> BaseTracker:
    GlobalHydra.instance().clear()
    initialize(version_base='1.3.2', config_path='config', job_name="multiscale")
    return BaseTracker(gpu, compose(config_name=f"gui_config"))


def subproc_main(gpu  = 'cuda:0', res = 720):
    r = redis.StrictRedis(connection_pool=pool)
    pubsub = r.pubsub()
    pubsub.subscribe(FRAME_CHANNEL)
    tracker = None
    for message in pubsub.listen():
        # print(gpu, end=' ')
        if message['type'] == 'message':
            data = pickle.loads(message['data'])
            if data['status'] == 'start':
                tracker = init_one_tracker(gpu)
                tracker.tracker.max_internal_size = res
                tracker.track(data['img'], data['mask'])
                continue
            elif data['status'] == 'end':
                r.close()
                return
            else:
                _, prob, _ = tracker.track(data['img'])
            result = {'prob': prob}
            data = pickle.dumps(result)
            r.publish(RESULT_CHANNEL, data)


@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_multi_config.yaml')
def eval_vos(cfg: DictConfig):
    if cfg['output_dir'] is not None:
        run_dir = cfg['output_dir']
    else:
        run_dir = HydraConfig.get().run.dir
    log.info(f'All configuration: {cfg}')

    # Load the network weights
    cutie = CUTIE(cfg).cuda(5).eval()
    if cfg.weights is not None:
        model_weights = torch.load(cfg.weights)
        cutie.load_weights(model_weights)
    else:
        log.warning('No model weights loaded. Are you sure about this?')

    dataset_name = cfg.dataset
    data_cfg = get_dataset_cfg(cfg)
    is_burst = ('burst' in dataset_name)

    # setup dataset
    image_dir = data_cfg.image_directory
    json_dir = data_cfg.get('json_directory')
    size_dir = data_cfg.get('size_directory')
    if is_burst:
        # BURST style -- masks stored in a json file
        meta_dataset = BURSTTestDataset(image_dir,
                                        json_dir,
                                        size=data_cfg.size,
                                        skip_frames=data_cfg.skip_frames)
        burst_handler = BURSTResultHandler(meta_dataset.json)
    else:
        # DAVIS/YouTubeVOS/MOSE style -- masks stored as PNGs
        mask_dir = data_cfg.mask_directory
        subset = data_cfg.get('subset')
        meta_dataset = VOSTestDataset(image_dir,
                                      mask_dir,
                                      use_all_masks=data_cfg.use_all_masks,
                                      req_frames_json=json_dir,
                                    #   size=data_cfg.size,
                                      size_dir=size_dir,
                                      subset=subset)
    use_amp = cfg.amp

    # multi-scale configurations
    save_scores = cfg.save_scores
    save_aux = cfg.save_aux
    if save_aux:
        # should have no effects on any other modules except on aux output
        cutie = cutie.train()

    # Set up loader
    meta_loader = meta_dataset.get_datasets()

    # determine where to save the masks
    save_all = data_cfg['save_all']
    mask_output_root = path.join(run_dir, 'Annotations')
    score_output_root = path.join(run_dir, 'Scores')
    visualize_output_root = path.join(run_dir, 'Visualizations')

    total_process_time = 0
    total_frames = 0

    r = redis.StrictRedis(connection_pool=pool)
    pubsub = r.pubsub()
    pubsub.subscribe(RESULT_CHANNEL)


    # Start eval
    pbar = tqdm(meta_loader, total=len(meta_dataset))
    for vid_reader in pbar:

        loader = DataLoader(vid_reader, batch_size=None, shuffle=False, num_workers=4)
        vid_name = vid_reader.vid_name
        pbar.set_description(vid_name)
        vid_length = len(loader)

        try:
            processor = InferenceCore(cutie, cfg=cfg)
            processor.max_internal_size = 800
            saver = ResultSaver(mask_output_root,
                                vid_name,
                                dataset=dataset_name,
                                object_manager=processor.object_manager,
                                use_long_id=vid_reader.use_long_id,
                                palette=vid_reader.get_palette(),
                                save_scores=save_scores,
                                score_output_root=score_output_root,
                                visualize_output_root=visualize_output_root,
                                visualize=cfg.visualize,
                                init_json=vid_reader.sequence_json if is_burst else None)
            first_mask_loaded = False

            # InstanceSegmenter = InstanceSegmenter()




            for ti, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    image = data['rgb'].cuda()
                    mask = data.get('mask')
                    if mask is not None:
                        mask = mask.cuda()
                    valid_labels = data.get('valid_labels')
                    if valid_labels is not None:
                        valid_labels = valid_labels.tolist()
                    info = data['info']
                    frame_name = info['frame']
                    shape = info['shape']
                    # print(shape)
                    resize_needed = info['resize_needed']
                    path_to_image = info['path_to_image']

                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                    # if, for some reason, the first frame is not aligned with the first mask
                    if not first_mask_loaded:
                        if mask is not None:
                            first_mask_loaded = True
                        else:
                            # no point to do anything without a mask
                            continue
                    if ti == 0:
                        data = pickle.dumps({'status': 'start', 'img': image.permute(1,2,0).cpu().numpy(), 'mask': mask.cpu().numpy()})
                    else:
                        data = pickle.dumps({'status': 'running','img': image.permute(1,2,0).cpu().numpy()})
                    r.publish(FRAME_CHANNEL, data)   

                    # Run the model on this frame
                    if mask is not None:
                        mask = mask.cuda(5)
                    
                    prob = processor.step(image.cuda(5), mask, valid_labels, end = ti == vid_length - 1).cpu()


                    if ti != 0:
                        cnt = 0
                        while cnt < 2:
                            message = pubsub.get_message()
                            while message is None:
                                message = pubsub.get_message()
                            if message['type'] != 'message':
                                continue
                            message_content = pickle.loads(message['data'])
                            # print(message_content['prob'].shape, prob.shape)
                            prob += message_content['prob'].cpu()
                            cnt+=1
                        prob = prob / 3


                    end.record()
                    torch.cuda.synchronize()
                    total_process_time += (start.elapsed_time(end) / 1000)
                    total_frames += 1

                    if save_all or info['save']:
                        saver.process(prob,
                                      frame_name,
                                      resize_needed=resize_needed,
                                      shape=shape,
                                      last_frame=(ti == vid_length - 1),
                                      path_to_image=path_to_image)

            saver.end()
            if is_burst:
                burst_handler.add_sequence(saver.video_json)

        except Exception as e:
            log.error(f'Runtime error at {vid_name}')
            log.error(e)
            saver.end()
            raise e

    log.info(f'Total processing time: {total_process_time}')
    log.info(f'Total processed frames: {total_frames}')
    log.info(f'FPS: {total_frames / total_process_time}')
    log.info(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

    make_zip(dataset_name, run_dir, cfg.exp_id, mask_output_root)
    if is_burst:
        burst_handler.dump(run_dir)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main_proc = mp.Process(target=eval_vos)
    p1 = mp.Process(target=subproc_main, args=('cuda:7', 600))
    p2 = mp.Process(target=subproc_main, args=('cuda:6', 720))

    p1.start()
    p2.start()
    main_proc.start()

    main_proc.join()
    p1.join()
    p2.join()
