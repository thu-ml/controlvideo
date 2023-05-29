# ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing 
This is the official implementation for "[ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing](http://arxiv.org/abs/2305.17098)". The project page is available [here](https://ml.cs.tsinghua.edu.cn/controlvideo/). Code will be released soon.
## Overview
ControlVideo incorporates visual conditions for all frames to amplify the source video's guidance, key-frame attention that aligns all frames with a selected one and temporal attention modules succeeded by a zero convolutional layer for temporal consistency and faithfulness. The three key components and corresponding fine-tuned parameters are designed by a systematic empirical study. Built upon the trained ControlVideo, during inference, we employ DDIM inversion and then generate the edited video using the target prompt via DDIM sampling.
![image](data/method.png)
## Main Results
![image](data/demo.png)

