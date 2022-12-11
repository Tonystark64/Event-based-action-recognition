# Event-based-action-recognition
Action Recognition with event-based camera data in Jupyter notebook   
The code is forked and modified from MMACTION TSM video understanding  
https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md  
We reconstructed event-based data to frames and apply TSM to event-generated frames, grayscale frames and RGB frames.  
With data augmentation skills, the accuracy of action recognition can be on par with grayscale frames, or even RGB frames.  
We also investigated how the event-frame should be generated and how to augment grayscale frames with event points to achieve improvements.  

[1] Lin, Ji and Gan, Chuang and Han, Song, "TSM: Temporal Shift Module for Efficient Video Understanding", in Proceedings of the IEEE International Conference on Computer Vision, 2019, lin2019tsm   
[2] Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He, "Non-local Neural Networks", CVPR, 2018, NonLocal2018
