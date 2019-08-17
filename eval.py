import glob
import os.path as osp
import skimage
import numpy as np
import cv2
import os
import h5py
import time
import sys

from model import CTPN
import utils
import config
from data_generator import data_generator
from split import show_image
from text_connector.detectors import TextDetector


def evaluate():
    for image_path in image_paths:
        image = cv2.imread(image_path)[:, :, ::-1]
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]
        result_label_fname = 'res_' + image_fname_noext + '.txt'
        result_label_path = osp.join(result_label_dir, result_label_fname)
        h, w = image.shape[:2]
        src_image = image.copy()
        image, scale, pad, window = utils.resize_and_pad_image(image, 512)
        image = utils.mold_image(image)
        image = np.expand_dims(image, axis=0)
        # Run object detection
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
        text_boxes = text_boxes[:, [0, 1, 4, 5]]
        # -left_pad
        text_boxes[:, [0, 2]] -= pad[1][0]
        # -top_pad
        text_boxes[:, [1, 3]] -= pad[0][0]
        text_boxes = np.round(text_boxes / scale).astype(np.int32)
        text_boxes[:, [0, 2]] = np.clip(text_boxes[:, [0, 2]], 0, w - 1)
        text_boxes[:, [1, 3]] = np.clip(text_boxes[:, [1, 3]], 0, h - 1)
        with open(result_label_path, 'w') as f:
            for text_box in text_boxes:
                f.write(','.join(map(str, text_box.tolist())) + '\n')
        # for box in text_boxes:
        #     cv2.rectangle(src_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        # show_image(src_image, 'image')
        # cv2.waitKey(0)


if __name__ == '__main__':
    weights_path = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    inference_mode = 'rpn'
    # Create model
    model = CTPN(mode="inference", inference_mode=inference_mode)
    # weights_path = 'ctpn/checkpoints/2019-07-24/ctpn_116-0.2168-0.3804.h5'
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
    image_paths = glob.glob('/home/adam/.keras/datasets/icdar2013/focused_scene_text/task12_test_images/*.jpg')
    result_label_dir = 'ctpn/evaluation/icdar2013'
    evaluate()
