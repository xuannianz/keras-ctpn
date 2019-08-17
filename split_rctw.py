import cv2
import glob
import json
import logging
import numpy as np
import os.path as osp
import sys


def resize_and_pad_image(image, input_size):
    """
    resize 并且 pad 使图像的大小为 input_size, 且保持 ratio 不变
    Args:
        image:
        input_size:

    Returns:
        padded_image:
        scale:
        pad:
        window:

    """
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    pad_top = (input_size - new_h) // 2
    pad_bottom = input_size - new_h - pad_top
    pad_left = (input_size - new_w) // 2
    pad_right = input_size - new_w - pad_left
    pad = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
    padded_image = np.pad(resized_image, pad, mode='constant', constant_values=0)
    return padded_image, scale, pad, (pad_top, pad_left, new_h + pad_top, new_w + pad_left)


def resize_cnts(cnts, scale, pad, window):
    resized_cnts = []
    for cnt in cnts:
        cnt = np.round(cnt * scale).astype(np.int32)
        # + pad_left
        cnt[:, 0] += pad[1][0]
        # + pad top
        cnt[:, 1] += pad[0][0]
        cnt[:, 0] = np.clip(cnt[:, 0], window[1], window[3])
        cnt[:, 1] = np.clip(cnt[:, 1], window[0], window[2])
        resized_cnts.append(cnt)
    return resized_cnts


def show_image(image, name, contours=None):
    image = image.astype(np.uint8)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if contours is not None:
        if isinstance(contours, list):
            cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        else:
            cv2.drawContours(image, [contours], -1, (0, 0, 255), 2)
    cv2.imshow(name, image)


def split_mask_to_boxes(cnt_mask, split_w=16):
    h, w = cnt_mask.shape
    extra = w % split_w
    if extra > 8:
        num = w // split_w + 1
    else:
        num = w // split_w

    cnt_boxes = []

    if num <= 2:
        return cnt_boxes

    for i in range(num):
        xmin, ymin, xmax, ymax = split_w * i, 0, split_w * (i + 1), h
        cnt_box_mask = cnt_mask[ymin: ymax, xmin: xmax]
        # 迭代寻找最优 ymin, ymax
        find_ymin = False
        find_ymax = False
        for j in range(ymax):
            if not find_ymin:
                if cnt_box_mask[j].max():
                    ymin = j
                    find_ymin = True
            if not find_ymax:
                if cnt_box_mask[ymax - j - 1].max():
                    ymax = ymax - j
                    find_ymax = True
        cnt_boxes.append([ymin, xmin, ymax, xmax])
    return cnt_boxes


if __name__ == '__main__':
    logger = logging.getLogger('build_dataset')
    logger.setLevel(logging.DEBUG)  # default log level
    formatter = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
    sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    quad_label_dir = 'datasets/rctw/train_quad_labels'
    label_dir = 'datasets/rctw/train_labels'
    for idx, image_path in enumerate(glob.glob('datasets/rctw/train_images/*.jpg')):
        logger.debug('Handling {}'.format(image_path))
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]
        label_fname = image_fname_noext + '.txt'
        label_path = osp.join(label_dir, label_fname)
        quad_label_path = osp.join(quad_label_dir, label_fname)
        cnts = []
        with open(quad_label_path) as f:
            lines = f.readlines()
            for line in lines:
                parts = [int(x) for x in line.strip().strip('\ufeff').split(',')[:9]]
                points = np.array(parts[:8]).reshape(4, 2)
                ignore = parts[-1]
                if not ignore:
                    cnts.append(points)
        if len(cnts) == 0:
            logger.error('{} has no labels'.format(image_path))
            continue
        image = cv2.imread(image_path)
        # show_image(image, 'image', cnts)
        # cv2.waitKey(0)
        image, scale, pad, window = resize_and_pad_image(image, 512)
        cnts = resize_cnts(cnts, scale, pad, window)
        # show_image(image, 'image', cnts)
        # cv2.waitKey(0)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, cnts, 255)
        # show_image(mask, 'mask')
        # cv2.waitKey(0)
        boxes = []
        for cnt in cnts:
            min_x = np.min(cnt[:, 0])
            max_x = np.max(cnt[:, 0])
            min_y = np.min(cnt[:, 1])
            max_y = np.max(cnt[:, 1])
            cnt_mask = mask[min_y: max_y, min_x: max_x]
            cnt_boxes = split_mask_to_boxes(cnt_mask)
            if len(cnt_boxes) == 0:
                continue
            cnt_boxes = np.array(cnt_boxes)
            cnt_boxes[:, [0, 2]] = cnt_boxes[:, [0, 2]] + min_y
            cnt_boxes[:, [1, 3]] = cnt_boxes[:, [1, 3]] + min_x
            boxes.extend(cnt_boxes.tolist())
        if len(boxes) == 0:
            logger.error('{} has no labels after resize'.format(image_path))
            continue
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(','.join(map(str, box)) + '\n')

        # for box in boxes:
        #     cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 1)
        # show_image(image, 'image')
        # cv2.waitKey(0)
