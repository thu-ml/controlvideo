import os
from util import available_devices, format_devices
device = available_devices(threshold=40000, n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
from omegaconf import OmegaConf
from train import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--control_type",
        type=str,
        default='hed',
        help="the type of control"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default='videos/car10.mp4',
        help="the path to the input video"
    )

    parser.add_argument(
        "--source",
        type=str,
        default='a car',
        help="the prompt for source video"
    )

    parser.add_argument(
        "--target",
        type=str,
        default='a red car',
        help="the prompt for target video"
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default='outputs/',
        help="the path for saving"
    )

    parser.add_argument(
        "--max_step",
        type=int,
        default=300,
        help="the steps for training"
    )

    args = parser.parse_args()

    name = args.video_path.split('/')[-1]
    name = name.split('.')[0]
    config_root = "./configs/default/"
    config = os.path.join(config_root, f"{args.control_type}.yaml")
    para = OmegaConf.load(config)
    para.train_data.video_path = args.video_path
    para.output_dir = os.path.join(args.out_root, f"{name}-{args.target}")
    para.train_data.prompt = args.source
    para.validation_data.prompts = [args.target]
    para.max_train_steps = args.max_step
    para.validation_steps = para.max_train_steps # the validation_steps are set to max_train_steps for saving time

    main(**para)







