import os, glob

root = '/home/gaomingqi/unify/MOSE/valid/JPEGImages'
video_list = glob.glob(os.path.join(root, '*'))
for vid in video_list:
    frame_list = glob.glob(os.path.join(vid, '.*'))
    # remove_list = [f for f in frame_list if '._' in f]
    for r in frame_list:
        os.remove(r)


root = '/home/gaomingqi/unify/MOSE/valid/Annotations'
video_list = glob.glob(os.path.join(root, '*'))
for vid in video_list:
    frame_list = glob.glob(os.path.join(vid, '.*'))
    # remove_list = [f for f in frame_list if '._' in f]
    for r in frame_list:
        os.remove(r)



root = '/home/gaomingqi/unify/MOSE/valid'

