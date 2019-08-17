import tensorflow as tf
import keras.backend as K
import utils_graph
import config


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """
    Implements Smooth-L1 loss.

    Args:
        y_true: here is [N, 4], but could be any shape.
        y_pred: here is [N, 4], but could be any shape.

    Returns:

    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    RPN anchor classifier loss.

    num_anchors = feature_map_height * feature_map_width * num_anchor_scales * num_anchor_ratios

    Args:
        rpn_match: (batch_size, num_anchors, 1). Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
            真值
        rpn_class_logits: (batch_size, num_anchors, 2). RPN classifier logits for BG/FG.
            rpn_model 输出的值

    Returns:

    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    # pos anchor 对应的值设为 1, neutral 和 neg anchor 对应的值设为 0
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    # 非 neutral 的 anchors 的 indices
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    # Note: anchor_class 是包含 neutral anchor 的, 现在只获取 pos anchor 和 neg anchor 的值
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    # 使用 sparse_categorical_crossentropy, anchor_class 不需要是 one-hot 形式
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(target_rpn_bbox, target_rpn_match, pred_rpn_bbox):
    """
    Return the RPN bounding box loss graph.

    Args:
        target_rpn_bbox: [batch, RPN_TRAIN_ANCHORS_PER_IMAGE, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unused bbox deltas.
            非 pos anchors 的部分都是 0
        target_rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
        pred_rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
            预测值
    Returns:

    """
    # Positive anchors contribute to the loss, but negative and neutral anchors (match value of 0 or -1) don't.
    target_rpn_match = K.squeeze(target_rpn_match, -1)
    indices = tf.where(K.equal(target_rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    pred_rpn_bbox = tf.gather_nd(pred_rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    # 每一个 batch 中 pos anchors 的个数
    batch_counts = K.sum(K.cast(K.equal(target_rpn_match, 1), tf.int32), axis=1)
    # 把每一个 batch 的 pos anchors 组合起来
    target_rpn_bbox = utils_graph.batch_pack_graph(target_rpn_bbox, batch_counts, config.BATCH_SIZE)
    loss = smooth_l1_loss(target_rpn_bbox, pred_rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss
