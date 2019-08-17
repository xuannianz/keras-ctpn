import h5py
import numpy as np
import glob
import cv2
import os.path as osp
import sys
from tqdm import tqdm
import logging
import random
import shutil
import os

from utils import resize_and_pad_image, mold_image, unmold_image
import config


def get_dataset_size(data_dir):
    image_paths = glob.glob(osp.join(data_dir, '**/*.jpg'), recursive=True)
    return len(image_paths)


def delete_empty_labels_and_images():
    label_paths = glob.glob(osp.join(label_dir, '*.txt'))
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                logger.warning('{} is empty'.format(label_path))
                label_fname = osp.split(label_path)[-1]
                label_fname_noext = osp.splitext(label_fname)[0]
                image_fname = label_fname_noext + '.jpg'
                image_path = osp.join(image_dir, image_fname)
                os.remove(label_path)
                os.remove(image_path)


def delete_images_without_labels():
    image_paths = glob.glob(osp.join(image_dir, '*.jpg'))
    for image_path in image_paths:
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]

        label_fname = image_fname_noext + '.txt'
        label_path = osp.join(label_dir, label_fname)
        if not osp.exists(label_path):
            logger.warning('{} does not exist'.format(label_path))
            os.remove(image_path)


def create_hdf5(image_paths, h5_path, input_size=config.RPN_INPUT_SIZE):
    if osp.exists(h5_path):
        logger.warning('{} already exist'.format(h5_path))
        return
    hdf5_dataset = h5py.File(h5_path, 'w')
    dataset_size = len(image_paths)
    hdf5_images = hdf5_dataset.create_dataset(name='images',
                                              shape=(dataset_size, config.RPN_INPUT_SIZE, config.RPN_INPUT_SIZE, 3),
                                              dtype=np.float32)
    hdf5_images.attrs['dataset_size'] = dataset_size
    hdf5_image_fnames = hdf5_dataset.create_dataset(name='image_fnames',
                                                    shape=(dataset_size,),
                                                    dtype=h5py.special_dtype(vlen=str))
    hdf5_gt_boxes = hdf5_dataset.create_dataset(name='gt_boxes',
                                                shape=(dataset_size,),
                                                dtype=h5py.special_dtype(vlen=np.int32))
    # Create the dataset that will hold the dimensions of the gt_boxes for each image so that we can
    # restore the gt_boxes from the flattened arrays later.
    hdf5_num_gt_boxes = hdf5_dataset.create_dataset(name='num_gt_boxes',
                                                    shape=(dataset_size,),
                                                    dtype=np.int32)
    for idx, image_path in enumerate(tqdm(image_paths, 'Creating hdf5 dataset', file=sys.stdout)):
        image = cv2.imread(image_path)
        image, scale, pad, window = resize_and_pad_image(image, input_size)
        # mold_image 传入 RGB 图像, 同样也返回 RGB
        image = mold_image(image[:, :, ::-1])
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]

        label_fname = image_fname_noext + '.txt'
        label_path = osp.join(label_dir, label_fname)
        gt_boxes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                gt_box = list(map(int, line.strip().split(',')))
                gt_boxes.append(gt_box)
        if len(gt_boxes) == 0:
            logger.error('no gt_boxes of {}'.format(image_path))
            break
        gt_boxes = np.array(gt_boxes)
        num_gt_boxes = gt_boxes.shape[0]

        hdf5_images[idx] = image
        hdf5_image_fnames[idx] = image_fname_noext
        hdf5_num_gt_boxes[idx] = num_gt_boxes
        hdf5_gt_boxes[idx] = gt_boxes.reshape((-1,))

    logger.debug('Writing {} instances done!!!'.format(idx + 1))
    hdf5_dataset.close()


def show_image(image, name, contours=None):
    image = image.astype(np.uint8)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if contours is not None:
        if isinstance(contours, list):
            cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        else:
            cv2.drawContours(image, [contours], -1, (0, 0, 255), 2)
    cv2.imshow(name, image)


def draw_rect(image, window_name, locations=None):
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (0, 255, 0), 4)
    for start_y, start_x, end_y, end_x in locations:
        color = np.random.randint(255, size=3).tolist()
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
    cv2.namedWindow('{}'.format(window_name), cv2.WINDOW_NORMAL)
    cv2.imshow('{}'.format(window_name), image)


def test_hdf5(h5_path):
    hdf5_dataset = h5py.File(h5_path, 'r')
    hdf5_images = hdf5_dataset['images']
    dataset_size = len(hdf5_images)
    hdf5_gt_boxes = hdf5_dataset['gt_boxes']
    hdf5_num_gt_boxes = hdf5_dataset['num_gt_boxes']
    for i in range(dataset_size):
        image = hdf5_images[i]
        image = unmold_image(image)[:, :, ::-1].astype(np.uint8)
        num_gt_boxes = hdf5_num_gt_boxes[i]
        gt_boxes = hdf5_gt_boxes[i].reshape((num_gt_boxes, 4))
        draw_rect(image, 'image', gt_boxes.tolist())
        cv2.waitKey(0)


if __name__ == '__main__':
    logger = logging.getLogger('build_dataset')
    logger.setLevel(logging.DEBUG)  # default log level
    formatter = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
    sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    image_paths = []
    art_data_dir = 'datasets/art'
    rctw_data_dir = 'datasets/rctw'
    for data_dir in [art_data_dir, rctw_data_dir]:
        image_dir = osp.join(art_data_dir, 'train_images')
        label_dir = osp.join(art_data_dir, 'train_labels')
        label_paths = glob.glob(osp.join(label_dir, '*.txt'))
        current_image_paths = [osp.join(image_dir, osp.split(label_path)[-1][:-4] + '.jpg') for label_path in label_paths]
        image_paths.extend(current_image_paths)
    random.shuffle(image_paths)
    num_images = len(image_paths)
    num_train_images = int(num_images * 0.8)
    num_train_images = num_train_images - num_train_images % config.BATCH_SIZE
    num_val_images = num_images - num_train_images
    num_val_images = num_val_images - num_val_images % config.BATCH_SIZE
    logger.debug(
        'num_images={}, num_train_images={}, num_val_images={}'.format(num_images, num_train_images, num_val_images))
    train_image_paths = image_paths[:num_train_images]
    val_image_paths = image_paths[num_train_images: num_train_images + num_val_images]

    # delete_empty_labels_and_images()
    # delete_images_without_labels()
    create_hdf5(train_image_paths, 'ctpn/train_art_rctw_0723_{}_{}.h5'.format(num_train_images, num_val_images))
    create_hdf5(val_image_paths, 'ctpn/val_art_rctw_0723_{}_{}.h5'.format(num_train_images, num_val_images))
    # test_hdf5('ctpn/val_art_rctw_0723_8964_2232.h5')
