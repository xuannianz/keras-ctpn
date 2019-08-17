import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import numpy as np

import utils
import utils_graph
import config


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas):
    """
    Refine classified proposals and filter overlaps and return final detections.
    Args:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            N = POST_NMS_ROIS_INFERENCE = 1000
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific bounding box deltas.

    Returns:
        detections: [num_detections, (y1, x1, y2, x2, class_id, score)] where coordinates are normalized.

    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    # (N, 2) 第二维第一个元素表示 roi 的 id, 第二个元素表示 roi 关联的 class_id
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    # (N, ) 每个 roi 最大的 score
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area
    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        # set_intersection 不能接受 1-d tensor, 所以 expand_dims 一下
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        # sparse_tensor_to_dense 返回 tensor 的 shape 为 (1, n)
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    # unique[0] 表示 pre_nms_class_ids 中 unique class id
    # unique[1] 表示 unique class id 在 pre_nms_class_ids 中的 indices
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    # 按 class_score 进行从大到小排序
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        rois = inputs[0]
        rcnn_class = inputs[1]
        rcnn_delta = inputs[2]

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice([rois, rcnn_class, rcnn_delta],
                                             lambda x, y, w: refine_detections_graph(x, y, w),
                                             config.BATCH_SIZE)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(detections_batch, [config.BATCH_SIZE, config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return None, config.DETECTION_MAX_INSTANCES, 6


############################################################
#  Detection Target Layer
############################################################


def detection_targets_graph(proposals, gt_class_ids, gt_boxes):
    """
    Generates detection targets for one image. Subsamples proposals and generates target class IDs, bounding box deltas.

    计算 rpn 网络输出的 proposals 和所有 gt_boxes 的 iou, 如果 iou >= 0.5 认为是 pos roi, iou < 0.5, 认为是 neg roi.
    pos rois 和 neg rois 以 1:2 的数量组合, 如果两者加起来的数量还不足 POST_NMS_ROIS_TRAINING 个, 用 0 来填充.
    找到每个 pos roi 和其有最大 iou 的 gt_box, 计算 delta, 且该 gt_box 的 class_id 作为该 pos roi 的 class_id

    最后返回三个值: rois, class_ids, deltas

    Args:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates.
            Might be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
            Might be zero padded if there are not enough proposals.
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
            Might be zero padded if there are not enough proposals.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        Note: Returned arrays might be zero padded if not enough target ROIs.

    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = utils_graph.trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = utils_graph.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = utils_graph.overlaps_graph(proposals, gt_boxes)

    # Determine positive and negative ROIs
    # 每一个 roi(proposal) 和所有 gt_boxes 的最大 iou
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    # 每个 positive overlap 和所有 gt_boxes 的 overlap(iou)
    positive_overlaps = tf.gather(overlaps, positive_indices)
    # 如果没有 positive overlap, 返回一个空张量, 否则返回每个 positive overlap 对应的 gt_box_id
    roi_gt_box_assignment = tf.cond(
        # UNCLEAR: 难道不应该判断是否有 positive_overlaps, 而不应该判断是否有 gt_boxes 吗
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    # (num_pos_rois, 4)
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    # (num_pos_rois, )
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils_graph.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Append negative ROIs and pad bbox deltas that are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas


class DetectionTargetLayer(KE.Layer):
    """
    Subsamples proposals and generates target box refinement, class_ids.

    Inputs:
        proposals: [batch, config.POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates.
            Might be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
            Might be zero padded if there are not enough proposals.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
            Might be zero padded if there are not enough proposals.

    Returns:
        Target ROIs and corresponding class IDs, bounding box shifts.
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
              coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                     Masks cropped to bbox boundaries and resized to neural
                     network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = utils.batch_slice([proposals, gt_class_ids, gt_boxes],
                                    lambda x, y, z: detection_targets_graph(x, y, z),
                                    config.BATCH_SIZE, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """
    Receives anchor scores and selects a subset to pass as proposals to the second stage.
    Filtering is done based on anchor scores and non-max suppression to remove overlaps.
    It also applies bounding box refinement deltas to anchors.

    从 rpn_model 的输出中选择 pre_nms_limit 个 anchors, 然后根据 rpn_delta 对这些 anchors 进行修正, 最后再根据 rpn_scores 从这些
    修正好的元素中选择 proposal_count 个元素

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, mode, nms_threshold, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = config.POST_NMS_ROIS_TRAINING if mode == 'training' else config.POST_NMS_ROIS_INFERENCE
        self.nms_threshold = nms_threshold
        self.mode = mode

    def call(self, inputs, **kwargs):
        if self.mode == 'training':
            batch_size = config.BATCH_SIZE
        else:
            batch_size = 1
        # Box Scores. Use the foreground class confidence.
        # (batch_size, num_anchors), 这里的 1 表示的就是 foreground class confidence 的下标
        scores = inputs[0][:, :, 1]
        # Box deltas (batch_size, num_anchors, 4)
        deltas = inputs[1]
        deltas = deltas * np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Normalized anchors
        anchors = inputs[2]
        # Improve performance by trimming to top anchors by score and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # top_k 如果接受的是多维数组, 那么是对最后一维进行排序, 返回的 indices 和 values 的 shape 除了最后一维的长度为 k
        # 其他和 scores 保持一致
        # 所以这里 ix 的 shape 应该是 (batch_size, pre_nms_limit)
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        # scores 的 shape 变为 (batch_size, pre_nms_limit)
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), batch_size)
        # (batch_size, pre_nms_limit, 4)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), batch_size)
        # (batch_size, pre_nms_limit, 4)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda x, y: tf.gather(x, y), batch_size,
                                            names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                                  batch_size,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  batch_size,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count, self.nms_threshold,
                                                   name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            probs = tf.gather(scores, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            probs = tf.pad(probs, [(0, padding)])
            return proposals, probs

        # (batch_size, proposal_count, 4)
        proposals, probs = utils.batch_slice([boxes, scores], nms, batch_size)
        # 多个输出不能使用 tuple, 必须是 list
        return [proposals, probs]

    def compute_output_shape(self, input_shape):
        # 多个输出不能使用 tuple, 必须是 list
        return [[None, self.proposal_count, 4], [None, self.proposal_count]]


class FilterLayer(KE.Layer):
    """
    Receives anchor scores and selects a subset to pass as proposals to the second stage.
    Filtering is done based on anchor scores and non-max suppression to remove overlaps.
    It also applies bounding box refinement deltas to anchors.

    从 rpn_model 的输出中选择 pre_nms_limit 个 anchors, 然后根据 rpn_delta 对这些 anchors 进行修正, 最后再根据 rpn_scores 从这些
    修正好的元素中选择 proposal_count 个元素

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, text_min_score, text_nms_threshold, text_proposal_count, **kwargs):
        super(FilterLayer, self).__init__(**kwargs)
        self.text_min_score = text_min_score
        self.text_nms_threshold = text_nms_threshold
        self.text_proposal_count = text_proposal_count

    def call(self, inputs, **kwargs):
        # Box coordinates
        # (batch_size, num_post_nms_rois=1000, 4)
        boxes = inputs[0]
        # Box Scores
        # (batch_size, num_post_nms_rois=1000)
        scores = inputs[1]
        batch_size = boxes.shape[0]

        def filter_on_score(boxes, scores):
            boxes_count = tf.shape(boxes)[0]
            keep_ix = tf.where(scores > self.text_min_score)[:, 0]
            boxes = tf.gather(boxes, keep_ix)
            scores = tf.gather(scores, keep_ix)
            current_boxes_count = tf.shape(boxes)[0]
            delta_count = boxes_count - current_boxes_count
            boxes = tf.pad(boxes, [(0, delta_count), (0, 0)])
            scores = tf.pad(scores, [(0, delta_count)])
            return boxes, scores

        # scores 的 shape 变为 (batch_size, pre_nms_limit)
        boxes, scores = utils.batch_slice([boxes, scores], filter_on_score, batch_size)

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.text_proposal_count, self.text_nms_threshold,
                                                   name="text_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            probs = tf.gather(scores, indices)
            # Pad if needed
            padding = tf.maximum(self.text_proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            probs = tf.pad(probs, [(0, padding)])
            return proposals, probs

        # (batch_size, proposal_count, 4)
        proposals, probs = utils.batch_slice([boxes, scores], nms, batch_size)
        # 多个输出不能使用 tuple, 必须是 list
        return [proposals, probs]

    def compute_output_shape(self, input_shape):
        # 多个输出不能使用 tuple, 必须是 list
        return [[None, self.text_proposal_count, 4], [None, self.text_proposal_count]]


