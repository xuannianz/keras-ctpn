import logging
import numpy as np
import h5py
import cv2
import os.path as osp
import config
import utils


def build_rpn_targets(anchors, gt_boxes):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    Args:
        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.

    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max_anchors_per_image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4), dtype=np.float32)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).

    # 1. Set negative anchors first. They get overwritten below if a GT box is matched to them.

    # 每个 anchor 对应有最大 overlap 的 gt_box_id
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    # 每个 anchor 和所有 gt_boxes 的最大 overlap
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    # 如果 anchor 和 所有 gt_boxes 的 overlap 都小于 0.3, 设置该 anchor 为 neg
    rpn_match[(anchor_iou_max < 0.3)] = -1

    # 2. Set an anchor for each GT box (regardless of IoU value).

    # If multiple anchors have the same IoU match all of them
    # 竟然会出现一个 gt_box 和多个 anchor 有相同的 iou, [:, 0] 表示只获取这些 anchor 的 id
    # overlaps == np.max(overlaps, axis=0) 会进行 broadcast, 得到和 overlaps 相同 shape 的结果
    # In [8]: a=np.array([[1, 0, 3, 2],[2, 1, 0, 1]])
    # In [9]: a == np.max(a, axis=0)
    # Out[9]:
    # array([[False, False,  True,  True],
    #        [ True,  True, False, False]])
    # 得到的是和各个 gt_box 有最大 overlap 的 anchor 的 id,
    # NOTE: 一个 gt_box 可能和多个 anchors 有最大 overlap 这些 anchor 都认为是 pos
    #  一个 anchor 也可能和多个 gt_boxes 有最大 iou
    # argwhere 返回值的 shape 为 (n, len(overlaps.shape)), n 表示符合条件的个数, 第二维表示符合条件的值的下标
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1

    # 3. Set anchors with high overlap as positive.
    # 如果 anchor 和所有 gt_boxes 的最大的 overlap 大于 0.7, 设置该 anchor 为 pos
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    # pos 和 neg anchors 的个数加起来等于 config.RPN_TRAIN_ANCHORS_PER_IMAGE
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    # index into rpn_bbox
    ix = 0
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Note: 这里的 i 表示的是 pos anchor 在所有 anchors 中的 id
        # Closest gt box (it might have IoU < 0.7)
        # 取和此 anchor 有最大 iou 的 gt_box
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def data_generator(h5_path, batch_size=config.BATCH_SIZE, input_size=config.RPN_INPUT_SIZE, is_training=False):
    """
    A generator that returns images and corresponding target class ids, bounding box deltas.

    Args:
        h5_path:
        batch_size: How many images to return in each call
        input_size:
        is_training:

    Returns:
        Returns a Python generator. Upon calling next() on it, the generator returns two lists, inputs and outputs.
        The contents of the lists differs depending on the received arguments:
        inputs list:
            - images: [batch, H, W, C]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]

        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.

    """
    # Anchors
    # num_anchors = feature_map_size * feature_map_size * num_anchor_scales * num_anchor_ratios
    # [num_anchors, (y1, x1, y2, x2)]

    feature_map_size = input_size // config.RPN_DOWNSCALE
    anchors = utils.generate_anchors(config.RPN_ANCHOR_HEIGHTS,
                                     config.RPN_ANCHOR_WIDTHS,
                                     feature_map_size,
                                     config.RPN_DOWNSCALE,
                                     config.RPN_ANCHOR_STRIDE,
                                     )
    current_idx = 0
    hdf5_dataset = h5py.File(h5_path, 'r')
    hdf5_images = hdf5_dataset['images']
    dataset_size = len(hdf5_images)
    # hdf5_image_shapes = hdf5_dataset['image_shapes']
    hdf5_gt_boxes = hdf5_dataset['gt_boxes']
    hdf5_num_gt_boxes = hdf5_dataset['num_gt_boxes']
    indicies = np.arange(dataset_size)
    if is_training:
        np.random.shuffle(indicies)
    # batch item index
    b = 0
    # Keras requires a generator to run indefinitely.
    while True:
        if current_idx >= dataset_size:
            if is_training:
                np.random.shuffle(indicies)
            current_idx = 0
        if b == 0:
            # Init batch arrays
            batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=np.int32)
            batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=np.float32)
            batch_images = np.zeros((batch_size, input_size, input_size, 3), dtype=np.float32)
            # batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
        i = indicies[current_idx]
        image = hdf5_images[i]
        # image_shape = hdf5_image_shapes[i]
        # image = image.reshape(image_shape)
        num_gt_boxes = hdf5_num_gt_boxes[i]
        gt_boxes = hdf5_gt_boxes[i].reshape((num_gt_boxes, 4))
        # image, scale, pad, window = utils.resize_and_pad_image(image, input_size)
        # gt_boxes = np.round(gt_boxes * scale).astype(np.int32)
        # (y1, x1, y2, x2)
        # gt_boxes = gt_boxes[:, [1, 0, 3, 2]]
        # + pad_left
        # gt_boxes[:, [1, 3]] += pad[1][0]
        # + pad top
        # gt_boxes[:, [0, 2]] += pad[0][0]
        # gt_boxes[:, [0, 2]] = np.clip(gt_boxes[:, [0, 2]], window[0], window[2])
        # gt_boxes[:, [1, 3]] = np.clip(gt_boxes[:, [1, 3]], window[1], window[3])

        # for gt_box in gt_boxes:
        #     cv2.rectangle(image, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0, 255, 0), 1)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_boxes)

        # Add to batch
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = image
        # batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        b += 1
        if b == batch_size:
            # inputs = [batch_images, batch_rpn_match, batch_rpn_bbox, batch_gt_boxes]
            inputs = [batch_images, batch_rpn_match, batch_rpn_bbox]
            outputs = []
            yield inputs, outputs
            b = 0
        current_idx += 1


def data_generator_2(h5_path):
    """
    A generator that returns images and corresponding target class ids, bounding box deltas.

    Args:
        h5_path:

    Returns:
        Returns a Python generator. Upon calling next() on it, the generator returns two lists, inputs and outputs.
        The contents of the lists differs depending on the received arguments:
        inputs list:
            - images: [batch, H, W, C]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]

        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.

    """
    # Anchors
    # num_anchors = feature_map_size * feature_map_size * num_anchor_scales * num_anchor_ratios
    # [num_anchors, (y1, x1, y2, x2)]

    current_idx = 0
    hdf5_dataset = h5py.File(h5_path, 'r')
    hdf5_images = hdf5_dataset['images']
    dataset_size = len(hdf5_images)
    hdf5_image_shapes = hdf5_dataset['image_shapes']
    hdf5_gt_boxes = hdf5_dataset['gt_boxes']
    hdf5_num_gt_boxes = hdf5_dataset['num_gt_boxes']
    indicies = np.arange(dataset_size)
    # Keras requires a generator to run indefinitely.
    while True:
        if current_idx >= dataset_size:
            np.random.shuffle(indicies)
            current_idx = 0
        i = indicies[current_idx]
        image = hdf5_images[i]
        image_shape = hdf5_image_shapes[i]
        image = image.reshape(image_shape)
        image, pad, window = utils.pad_image(image, config.RPN_DOWNSCALE)
        pad_image_shape = image.shape
        num_gt_boxes = hdf5_num_gt_boxes[i]
        gt_boxes = hdf5_gt_boxes[i].reshape((num_gt_boxes, 4))
        # (y1, x1, y2, x2)
        gt_boxes = gt_boxes[:, [1, 0, 3, 2]]
        # + pad_left
        gt_boxes[:, [1, 3]] += pad[1][0]
        # + pad top
        gt_boxes[:, [0, 2]] += pad[0][0]
        gt_boxes[:, [0, 2]] = np.clip(gt_boxes[:, [0, 2]], window[0], window[2])
        gt_boxes[:, [1, 3]] = np.clip(gt_boxes[:, [1, 3]], window[1], window[3])

        for gt_box in gt_boxes:
            cv2.rectangle(image, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0, 255, 0), 1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey(0)

        # RPN Targets
        anchors = utils.generate_anchors_2(config.RPN_ANCHOR_HEIGHTS,
                                           config.RPN_ANCHOR_WIDTHS,
                                           pad_image_shape[0] // config.RPN_DOWNSCALE,
                                           pad_image_shape[1] // config.RPN_DOWNSCALE,
                                           config.RPN_DOWNSCALE,
                                           )
        rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_boxes)

        # Add to batch
        batch_rpn_match = np.array([rpn_match[:, np.newaxis]])
        batch_rpn_bbox = np.array([rpn_bbox])
        batch_images = np.array([utils.mold_image(image.astype(np.float32)[:, :, ::-1])])
        batch_gt_boxes = np.array([gt_boxes])
        inputs = [batch_images, batch_rpn_match, batch_rpn_bbox, batch_gt_boxes]
        outputs = []
        yield inputs, outputs
        current_idx += 1


def show_anchors_before_rectify(image, rpn_match, anchors, gt_boxes):
    pos_anchor_ids = np.where(rpn_match[:, 0] == 1)[0]
    pos_anchors = anchors[pos_anchor_ids]
    if len(pos_anchors) > 10:
        pos_anchors = pos_anchors[:10]
    pos_anchors = np.round(pos_anchors).astype(np.int32)
    for pos_anchor in pos_anchors:
        cv2.rectangle(image, (pos_anchor[1], pos_anchor[0]), (pos_anchor[3], pos_anchor[2]), (0, 0, 255), 1)
    for gt_box in gt_boxes:
        cv2.rectangle(image, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0, 255, 0), 1)
    cv2.namedWindow('before', cv2.WINDOW_NORMAL)
    cv2.imshow('before', image)
    cv2.waitKey(0)


def show_anchors_after_rectify(image, rpn_match, rpn_bbox, anchors, gt_boxes):
    pos_anchor_ids = np.where(rpn_match[:, 0] == 1)
    pos_anchors = anchors[pos_anchor_ids]
    if len(pos_anchors) > 10:
        pos_anchors = pos_anchors[:10]
    rpn_bbox = rpn_bbox[:len(pos_anchors)]
    rpn_bbox *= config.RPN_BBOX_STD_DEV
    a_w = pos_anchors[:, 3] - pos_anchors[:, 1]
    a_h = pos_anchors[:, 2] - pos_anchors[:, 0]
    a_cx = (pos_anchors[:, 3] + pos_anchors[:, 1]) / 2
    a_cy = (pos_anchors[:, 2] + pos_anchors[:, 0]) / 2
    cy = rpn_bbox[:, 0] * a_h + a_cy
    cx = rpn_bbox[:, 1] * a_w + a_cx
    h = np.exp(rpn_bbox[:, 2]) * a_h
    w = np.exp(rpn_bbox[:, 3]) * a_w
    y1 = cy - h / 2
    x1 = cx - w / 2
    y2 = cy + h / 2
    x2 = cx + w / 2
    rectified_anchors = np.stack([y1, x1, y2, x2], axis=-1)
    rectified_anchors = np.round(rectified_anchors).astype(np.int32)
    for rectified_anchor in rectified_anchors:
        cv2.rectangle(image, (rectified_anchor[1], rectified_anchor[0]), (rectified_anchor[3], rectified_anchor[2]),
                      (0, 0, 255), 3)
    for gt_box in gt_boxes:
        cv2.rectangle(image, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0, 255, 0), 1)
    cv2.namedWindow('after', cv2.WINDOW_NORMAL)
    cv2.imshow('after', image)
    cv2.waitKey(0)


def verify_label(h5_path):
    hdf5_dataset = h5py.File(h5_path, 'r')
    hdf5_gt_boxes = hdf5_dataset['gt_boxes']
    hdf5_num_gt_boxes = hdf5_dataset['num_gt_boxes']
    for i in range(len(hdf5_num_gt_boxes)):
        num_gt_boxes = hdf5_num_gt_boxes[i]
        gt_boxes = hdf5_gt_boxes[i].reshape((num_gt_boxes, 4))
        w = gt_boxes[:, 3] - gt_boxes[:, 1]
        print(np.min(w))


if __name__ == '__main__':
    generator = data_generator('ctpn/train_art_0722_4482_1121.h5')
    anchors = utils.generate_anchors(config.RPN_ANCHOR_HEIGHTS,
                                     config.RPN_ANCHOR_WIDTHS,
                                     config.RPN_INPUT_SIZE // config.RPN_DOWNSCALE,
                                     config.RPN_DOWNSCALE,
                                     config.RPN_ANCHOR_STRIDE,
                                     )
    while True:
        # next(generator)
        batch_images, batch_rpn_match, batch_rpn_bbox, batch_gt_boxes = next(generator)[0]
        for i in range(len(batch_images)):
            image = batch_images[i]
            image = utils.unmold_image(image)[:, :, ::-1].astype(np.uint8)
            rpn_match = batch_rpn_match[i]
            rpn_bbox = batch_rpn_bbox[i]
            gt_boxes = batch_gt_boxes[i]
            # anchors = utils.generate_anchors_2(config.RPN_ANCHOR_HEIGHTS,
            #                                    config.RPN_ANCHOR_WIDTHS,
            #                                    image.shape[0] // config.RPN_DOWNSCALE,
            #                                    image.shape[1] // config.RPN_DOWNSCALE,
            #                                    config.RPN_DOWNSCALE,
            #                                    config.RPN_ANCHOR_STRIDE,
            #                                    )
            show_anchors_before_rectify(image.copy(), rpn_match, anchors, gt_boxes)
            show_anchors_after_rectify(image.copy(), rpn_match, rpn_bbox, anchors, gt_boxes)
        # image = batch_images[2]
        # image = utils.unmold_image(image)
        # rpn_match = batch_rpn_match[2]
        # rpn_bbox = batch_rpn_bbox[2]
        # gt_boxes = batch_gt_boxes[2]
        # show_anchors_before_rectify(image.copy(), rpn_match, anchors, gt_boxes)
    # verify_label('ctpn/val_art_0722_4482_1121.h5')
