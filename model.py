import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.applications.resnet50 import ResNet50

from layers import ProposalLayer, FilterLayer, BatchNorm
import config
import utils_graph
import utils
from losses import rpn_bbox_loss_graph, rpn_class_loss_graph


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # Shared convolutional base of the RPN
    # shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv_shared')(feature_map)
    # shared = KL.Dropout(0.5)(shared)
    shared = feature_map
    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(anchors_per_location * 2, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2], anchors is height * width * anchors_per_location
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


class CTPN:
    """
    Encapsulates the ctpn model functionality.
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, inference_mode='rpn'):
        """
        Args:
            mode: Either "training" or "inference"
        """
        assert mode in ['training', 'inference']
        assert inference_mode in ['rpn', 'text']
        self.mode = mode
        self.inference_mode = inference_mode
        self._anchor_cache = {}
        self.keras_model = self.build()

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        feature_map_size = image_shape[0] // config.RPN_DOWNSCALE
        # Cache anchors and reuse if image shape is the same
        if tuple(image_shape) not in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_anchors(
                config.RPN_ANCHOR_HEIGHTS,
                config.RPN_ANCHOR_WIDTHS,
                feature_map_size,
                config.RPN_DOWNSCALE,
                config.RPN_ANCHOR_STRIDE)
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def build(self):
        """
        Build ctpn architecture.

        """
        mode = self.mode
        # Inputs
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        # input_image = KL.Input(shape=[512, 512, config.IMAGE_SHAPE[2]], name="input_image")
        if mode == "training":
            # RPN GT
            # (batch_size, num_anchors, 1)
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            # (batch_size, RPN_TRAIN_ANCHORS_PER_IMAGE=256, 4)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Anchors
            # normalized anchors
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
            anchors = input_anchors

        # Build the shared convolutional layers.
        resnet50 = ResNet50(input_tensor=input_image, include_top=False)
        # (batch, height // 16, width // 16, 1024)
        x = resnet50.get_layer('activation_40').output
        x = KL.Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = KL.TimeDistributed(KL.Bidirectional(KL.LSTM(128, return_sequences=True)))(x)

        # RPN Model
        rpn = build_rpn_model(len(config.RPN_ANCHOR_HEIGHTS), depth=K.int_shape(x)[-1])
        rpn_class_logits, rpn_class, rpn_bbox = rpn([x])

        # Generate proposals
        # Proposals are [batch, proposal_count, (y1, x1, y2, x2)] in normalized coordinates and zero padded.
        rpn_proposals, rpn_probs = ProposalLayer(mode=mode,
                                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                                 name="ROI",
                                                 )([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(*x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])

            # Model
            inputs = [input_image, input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       rpn_proposals, rpn_class_loss, rpn_bbox_loss]
            model = KM.Model(inputs, outputs, name='ctpn')
        else:
            if self.inference_mode == 'rpn':
                model = KM.Model([input_image, input_anchors],
                                 [rpn_proposals, rpn_probs],
                                 name='ctpn')
            else:
                text_proposals, text_probs = FilterLayer(text_min_score=0.7,
                                                         text_nms_threshold=0.2,
                                                         text_proposal_count=200)([rpn_proposals, rpn_probs])
                model = KM.Model([input_image, input_anchors],
                                 [text_proposals, text_probs],
                                 name='ctpn')

        # model.summary()
        return model


if __name__ == '__main__':
    model = CTPN(mode='training')
