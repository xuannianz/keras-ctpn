import numpy as np
import tensorflow as tf
import cv2

import config


##################################################
# Batch Slicing
##################################################
# Some custom layers support a batch size of 1 only, and require a lot of work to support batches greater than 1.
# This function slices an input tensor across the batch dimension and feeds batches of size 1.
# Effectively, an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    Args:
        inputs: list of tensors. All must have the same first dimension length
        graph_fn: A function that returns a TF tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.

    Returns:

    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is a list of outputs to a list of outputs and each has
    # a list of slices
    # 比如 outputs 可能是 [[b_i_1_o1, b_i_1_o2],[b_i_2_o1, b_i_2_o2],[b_i_3_o1, b_i_3_o2],[b_i_4_o1, b_i_4_o2]]
    # 其中 b_i_1_o1 表示第一个 batch_item 的第一个 output
    # 重新组合成 [[b_i_1_o1, b_i_2_o1, b_i_3_o1, b_i_4_o1], [b_i_1_o2, b_i_2_o2, b_i_3_o2, b_i_4_o2]]
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)
    # stack 一下就变成 [batch_o1, batch_o2]
    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    # 如果只有一个输出, 返回 list 的第一个元素
    if len(result) == 1:
        result = result[0]

    return result


def compute_iou(box, boxes, box_area, boxes_area):
    """
    Calculates IoU of the given box with the array of the given boxes.

    Args:
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

    Returns:

    Note: the areas are passed in rather than calculated here for efficiency. Calculate once in the caller to avoid
     duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    Note: ssd 中有同时计算两组 boxes 的方法, 效率更高
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def generate_anchors(heights, widths, feature_map_size, feature_stride=16, anchor_stride=1):
    """

    Args:
        heights: 1D array of anchor heights in pixels. Example: (11, 16, 23, 33, 48, 68, 97, 139, 198, 283)
        widths: 1D array of anchor widths. Example: (16, 16, 16, 16, 16, 16, 16, 16, 16, 16)
        feature_map_size (int): spatial shape of the feature map over which to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
            就是 feature_map 上一个像素相当于原来图像上的尺寸, 也就是原图像到 feature_map 缩小的倍数
        anchor_stride: Stride of anchors on the feature map. For example, if the value is 2 then generate anchors for
            every other feature map pixel.
            anchor 的间隔(以 feature map 上的像素为单位), 如果是 1 就表示每个像素上都有 anchor, 如果是 2 表示每隔一个像素都会有 anchor

    Returns:

    """
    # (10, )
    heights = np.array(heights)
    widths = np.array(widths)

    # Enumerate shifts in feature space
    # 映射到原图像上的偏移量, Note 这个偏移量当做 anchor 的 center_x 和 center_y
    # (32=feature_map_size, )
    shifts_y = np.arange(0, feature_map_size, anchor_stride) * feature_stride
    # (32=feature_map_size, )
    shifts_x = np.arange(0, feature_map_size, anchor_stride) * feature_stride
    # feature_map 上每个像素的坐标偏移
    # (32, 32)
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    # shifts_x 的 ndims=2, 会先把它摊开成 1-d, (32*32, )
    # 那么 box_widths, box_centers_x 的 shape 就是 (32*32, 10)
    # box_width 为 [[16, 16, 16, 16, 16, 16, 16, 16, 16, 16], ... 32 * 32个]
    # box_centers_x 为 [[0,...10 个],[1,...10 个],...,[32, ...10 个], 重复 32 次]
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    # box_heights 为 [[11, 16, 23, 33, 48, 68, 97, 139, 198, 283], ... 32 * 32个]
    # box_centers_y 为 [[0,...10 个],重复 32 次, [1,...10 个], 重复 32 次, ... [32, ...10 个], 重复 32 次]
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    # (feature_map_size * feature_map_size * num_heights | num_widths, 2)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # (feature_map_size * feature_map_size * num_anchors_scales * num_anchor_ratios, 4)
    # Convert to corner coordinates (y1, x1, y2, x2)
    # (cy, cx) - 0.5 * (w, h) = (y1, x1)
    # (cy, cx) + 0.5 * (w, h) = (y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_anchors_2(heights, widths, feature_map_height, feature_map_width, feature_stride=16, anchor_stride=1):
    """

    Args:
        heights: 1D array of anchor heights in pixels. Example: (11, 16, 23, 33, 48, 68, 97, 139, 198, 283)
        widths: 1D array of anchor widths. Example: (16, 16, 16, 16, 16, 16, 16, 16, 16, 16)
        feature_map_height (int): spatial height of the feature map over which to generate anchors.
        feature_map_width (int): spatial width of the feature map over which to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
            就是 feature_map 上一个像素相当于原来图像上的尺寸, 也就是原图像到 feature_map 缩小的倍数
        anchor_stride: Stride of anchors on the feature map. For example, if the value is 2 then generate anchors for
            every other feature map pixel.
            anchor 的间隔(以 feature map 上的像素为单位), 如果是 1 就表示每个像素上都有 anchor, 如果是 2 表示每隔一个像素都会有 anchor

    Returns:

    """
    # (10, )
    heights = np.array(heights)
    widths = np.array(widths)

    # Enumerate shifts in feature space
    # 映射到原图像上的偏移量, Note 这个偏移量当做 anchor 的 center_x 和 center_y
    # (32=feature_map_size, )
    shifts_y = np.arange(0, feature_map_height, anchor_stride) * feature_stride
    # (32=feature_map_size, )
    shifts_x = np.arange(0, feature_map_width, anchor_stride) * feature_stride
    # feature_map 上每个像素的坐标偏移
    # (32, 32)
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    # shifts_x 的 ndims=2, 会先把它摊开成 1-d, (32*32, )
    # 那么 box_widths, box_centers_x 的 shape 就是 (32*32, 10)
    # box_width 为 [[16, 16, 16, 16, 16, 16, 16, 16, 16, 16], ... 32 * 32个]
    # box_centers_x 为 [[0,...10 个],[1,...10 个],...,[32, ...10 个], 重复 32 次]
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    # box_heights 为 [[11, 16, 23, 33, 48, 68, 97, 139, 198, 283], ... 32 * 32个]
    # box_centers_y 为 [[0,...10 个],重复 32 次, [1,...10 个], 重复 32 次, ... [32, ...10 个], 重复 32 次]
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    # (feature_map_size * feature_map_size * num_heights | num_widths, 2)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # (feature_map_size * feature_map_size * num_anchors_scales * num_anchor_ratios, 4)
    # Convert to corner coordinates (y1, x1, y2, x2)
    # (cy, cx) - 0.5 * (w, h) = (y1, x1)
    # (cy, cx) + 0.5 * (w, h) = (y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def mold_image(images):
    """Expects an RGB image (or array of images) and subtracts the mean pixel and converts it to float.
    Expects image colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


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


def pad_image(image, downscale):
    """
    pad 使图像的 h, w 都为 downscale 的整数倍
    Args:
        image:

    Returns:
        padded_image:
        pad:
        window:

    """
    h, w = image.shape[:2]
    pad_top = (downscale - h % downscale) // 2
    pad_bottom = (downscale - h % downscale) - pad_top
    pad_left = (downscale - w % downscale) // 2
    pad_right = (downscale - w % downscale) - pad_left
    pad = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    return padded_image, pad, (pad_top, pad_left, h + pad_top, w + pad_left)


def unmold_image(normalized_images):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
