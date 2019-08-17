import numpy as np

BATCH_SIZE = 12
BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
# Max number of final detections
DETECTION_MAX_INSTANCES = 100
# Minimum probability value to accept a detected instance
# ROIs below this threshold are skipped
DETECTION_MIN_CONFIDENCE = 0.7
# Non-maximum suppression threshold for detection
DETECTION_NMS_THRESHOLD = 0.3
# Size of the fully-connected layers in the classification graph
FPN_CLASSIFY_FC_LAYERS_SIZE = 1024
# Gradient norm clipping
GRADIENT_CLIP_NORM = 5.0
IMAGE_SHAPE = (512, 512, 3)
# Learning rate and momentum
# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
# weights to explode. Likely due to differences in optimizer
# implementation.
LEARNING_RATE = 0.001
LEARNING_MOMENTUM = 0.9
# Loss weights for more precise optimization.
# Can be used for R-CNN training setup.
LOSS_WEIGHTS = {
    "rpn_class_loss": 1.,
    "rpn_bbox_loss": 1.,
}
# Maximum number of ground truth instances to use in one image
MAX_GT_INSTANCES = 300
# Image mean (RGB)
MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
# Number of classification classes (including background)
NUM_CLASSES = 21
# ROIs kept after tf.nn.top_k and before non-maximum suppression
PRE_NMS_LIMIT = 6000
# Pooled ROIs
POOL_SIZE = 7
# ROIs kept after non-maximum suppression (training and inference)
POST_NMS_ROIS_TRAINING = 2000
POST_NMS_ROIS_INFERENCE = 1000
# Percent of positive ROIs used to train classifier
ROI_POSITIVE_RATIO = 0.33
RPN_INPUT_SIZE = 512
# 原图像到特征图尺寸缩小的倍数
RPN_DOWNSCALE = 16
# 10 个 anchors 的高和宽
RPN_ANCHOR_HEIGHTS = (11, 16, 23, 33, 48, 68, 97, 139, 198, 283)
RPN_ANCHOR_WIDTHS = (16, 16, 16, 16, 16, 16, 16, 16, 16, 16)
# Anchor stride
# If 1 then anchors are created for each cell in the backbone feature map.
# If 2, then anchors are created for every other cell, and so on.
RPN_ANCHOR_STRIDE = 1
# Non-max suppression threshold to filter RPN proposals.
# You can increase this during training to generate more proposals.
RPN_NMS_THRESHOLD = 0.7
# How many anchors per image to use for RPN training
RPN_TRAIN_ANCHORS_PER_IMAGE = 256
# Bounding box refinement standard deviation for RPN and final detections.
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
STEPS_PER_EPOCH = 747
# Size of the top-down layers used to build the feature pyramid
TOP_DOWN_PYRAMID_SIZE = 256
# Train or freeze batch normalization layers
#     None: Train BN layers. This is the normal mode
#     False: Freeze BN layers. Good when using a small batch size
#     True: (don't use). Set layer in training mode even when predicting
# Defaulting to False since batch size is often small
TRAIN_BN = None
# Number of ROIs per image to feed to classifier heads
# keep a positive:negative ratio of 1:3.
# You can increase the number of proposals by adjusting the RPN NMS threshold.
TRAIN_ROIS_PER_IMAGE = 200
VALIDATION_STEPS = 186
# Weight decay regularization
WEIGHT_DECAY = 0.00001
