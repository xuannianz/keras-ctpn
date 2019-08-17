# keras-ctpn
This is an implementation of [CTPN](https://arxiv.org/abs/1609.03605) on keras and Tensorflow. The project is based on [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
and [eragonruan/text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn). 
Thanks for their hard work. 
## Test
1. I trained by concatenating two different datasets (icdar 2019 art and rctw 2017). There are 8964 images for training and 2232 images for validation.
2. The best evaluation result is 

| recall | precision | hmean |
| ---- | ---- | ---- |
| 0.6886 | 0.8677 | 0.7678 |
3. Pretrained model is here. [baidu netdisk](https://pan.baidu.com/s/1iGhPXpmmUEzWYttYy6GMkg) extract code: ezcj     
4. `python3 inference.py` to test your image.  
## Train
### build dataset
* First you need to split the annotations into small bboxes whose width is 8px and save them into a txt file for each image. 
* `python3 split.py` to split annotations in icdar2019 art dataset.
* `python3 split_rctw.py` to split annotations in rctw 2017 dataset.
* `python3 build_dataset.py` to save the images and annotations into hdf5 files so that the model can be trained very fast.
### train
* `python3 train.py` to start training.
## Evaluate
* `python3 eval.py` to evaluate.
## Improve
* data augmentation
* deep backbone network
* fpn
