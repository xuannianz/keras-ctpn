import glob
import os.path as osp
import skimage
import numpy as np
import cv2
import os
import h5py
import time

from model import CTPN
import utils
import config
from data_generator import data_generator
from split import show_image
from text_connector.detectors import TextDetector

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
inference_mode = 'text'
# Create model
model = CTPN(mode="inference", inference_mode=inference_mode)
weights_path = 'ctpn.h5'
# Load weights
print("Loading weights ", weights_path)
model.keras_model.load_weights(weights_path, by_name=True)
# Anchors
anchors = model.get_anchors(config.IMAGE_SHAPE)
# Duplicate across the batch dimension because Keras requires it
# TODO: can this be optimized to avoid duplicating the anchors?
anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
# image_paths = ['/home/adam/Public/test.jpg']
# image_paths = glob.glob('datasets/art/train_images/*.jpg')
# image_paths = glob.glob('/home/adam/.keras/datasets/text/ctpn/VOCdevkit/VOC2007/JPEGImages/*.jpg')
image_paths = glob.glob('/home/adam/.keras/datasets/icdar2013/focused_scene_text/task12_images/*.jpg')
for image_path in image_paths:
    image = cv2.imread(image_path)[:, :, ::-1]
    image, scale, pad, window = utils.resize_and_pad_image(image, 512)
    src_image = image.copy()
    image = utils.mold_image(image)
    image = np.expand_dims(image, axis=0)
    # Run object detection
    start = time.time()
    batch_rpn_proposals, batch_rpn_probs = model.keras_model.predict([image, anchors], verbose=0)
    rpn_proposals = batch_rpn_proposals[0]
    rpn_probs = batch_rpn_probs[0]
    boxes = utils.denorm_boxes(rpn_proposals, (512, 512))
    scores = rpn_probs[..., np.newaxis]
    # keep_ix = np.where(rpn_probs > 0.7)[0]
    # boxes = boxes[keep_ix]
    # scores = rpn_probs[keep_ix]
    # for box in boxes:
    #     cv2.rectangle(src_image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 1)
    # show_image(src_image, 'image')
    # cv2.waitKey(0)
    detector = TextDetector()
    if inference_mode == 'rpn':
        text_boxes = detector.detect(boxes[:, [1, 0, 3, 2]].astype(np.float32), scores, (512, 512))
    # inference_mode == 'text'
    else:
        text_boxes = detector.detect2(boxes[:, [1, 0, 3, 2]].astype(np.float32), scores, (512, 512))
    end = time.time()
    print('time lapsed: {:.4f}'.format(end - start))
    for i, box in enumerate(text_boxes):
        cv2.polylines(src_image, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
    show_image(src_image, 'image')
    cv2.waitKey(0)
    # Process detections


