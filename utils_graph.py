import tensorflow as tf


def batch_pack_graph(x, counts, num_rows):
    """
    Picks different number of values from each row in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def box_refinement_graph(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.

    Args:
        box: [N, (y1, x1, y2, x2)]
        gt_box: [N, (y1, x1, y2, x2)]

    Returns:

    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def norm_boxes_graph(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.
     啥意思?

    Args:
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates

    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    # UNCLEAR: y2, x2 为什么要减去 1? 我知道有很多 gt_box, x2==w and y2==h
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def overlaps_graph(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.

    Args:
        boxes1: [m, (y1, x1, y2, x2)].
        boxes2: [n, (y1, x1, y2, x2)].

    Returns:

    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it using tf.tile() and tf.reshape.
    # Adam: 没有必要 expand_dims
    # b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    # (m, 4) --> (m, 4 * n) --> (n * m, 4)
    b1 = tf.reshape(tf.tile(boxes1, [1, tf.shape(boxes2)[0]]), [-1, 4])
    # (n, 4) --> (n * m, 4)
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    # (m, n)
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    Often boxes are represented with matrices of shape [N, 4] and are padded with zeros.
    This removes zero boxes.

    Args:
        boxes: [N, 4] matrix of boxes.
        name: name of tensor

    Returns:

    """
    # box 的 x1, y1, x2, y2 相加, 如果是 0 那么就是 False, 如果大于 0 那么就是 True
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros
