import jsonlines
import os
from util import available_devices, format_devices
device = available_devices(threshold=40000, n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)

video_list = 'demos.jsonl'
with jsonlines.open(video_list, 'r') as reader:
    videos = [video for video in reader]
reader.close()
for video in videos:
    control_type = video['type']
    video_path = os.path.join('videos', video['name']+'.mp4')
    source = video['source']
    target = video['target']
    max_step = video['step']
    command = 'python main.py --control_type %s --video_path %s --source "%s" --target "%s" --max_step %s' % (
    control_type, video_path, source, target[0], max_step)
    os.system(command)
