from data_generator import data_generator
import config
import utils
from model import CTPN
import os.path as osp
import os
import keras
import tensorflow as tf
from datetime import date

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
num_epoches = 200
# 8964
train_h5_path = osp.join('/home/adam/workspace/github/xuannianz/carrot/ctpn/train_art_rctw_0723_8964_2232.h5')
# 2232
val_h5_path = osp.join('/home/adam/workspace/github/xuannianz/carrot/ctpn/val_art_rctw_0723_8964_2232.h5')
train_generator = data_generator(train_h5_path, batch_size=config.BATCH_SIZE, is_training=True)
val_generator = data_generator(val_h5_path, batch_size=config.BATCH_SIZE)
today = date.today()
log_dir = 'logs/{}'.format(today)
# Create log_dir if it does not exist
if not osp.exists(log_dir):
    os.makedirs(log_dir)

checkpoints_dir = 'checkpoints/{}'.format(today)
if not osp.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
# Callbacks
callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir,
                                histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(
        '{}/ctpn_{{epoch:03d}}-{{loss:.4f}}-{{val_loss:.4f}}.h5'.format(checkpoints_dir),
        verbose=1, save_weights_only=True),
    keras.callbacks.CSVLogger(filename='{}/art.csv'.format(checkpoints_dir),
                              separator=',',
                              append=True)
]

model = CTPN(mode='training')


# model.keras_model.load_weights('ctpn/checkpoints/2019-07-24/ctpn_112-0.2316-0.3951.h5', by_name=True)
# model.keras_model.summary()

# Train
def compile(keras_model, lr=1e-4):
    """
    Gets the model ready for training. Adds losses, regularization, and
    metrics. Then calls the Keras compile() function.
    """
    # Optimizer object
    optimizer = keras.optimizers.Adam(lr=lr)
    # Add Losses
    # First, clear previously set losses to avoid duplication
    keras_model._losses = []
    keras_model._per_input_losses = {}
    loss_names = ["rpn_class_loss",
                  "rpn_bbox_loss",
                  ]
    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer.output in keras_model.losses:
            continue
        loss = (tf.reduce_mean(layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)

    # Add L2 Regularization
    reg_losses = [
        tf.constant(0.5) * keras.regularizers.l2(config.WEIGHT_DECAY)(w)
        for w in keras_model.trainable_weights]

    total_reg_loss = tf.add_n(reg_losses)
    keras_model.add_loss(total_reg_loss)

    # Compile
    keras_model.compile(optimizer=optimizer,
                        loss=[None] * len(keras_model.outputs))

    # Add metrics for losses
    for name in loss_names:
        if name in keras_model.metrics_names:
            continue
        layer = keras_model.get_layer(name)
        keras_model.metrics_names.append(name)
        loss = (tf.reduce_mean(layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.metrics_tensors.append(loss)

    keras_model.metrics_names.append('regularizer_loss')
    keras_model.metrics_tensors.append(total_reg_loss)


# trainable = False
# for layer in model.keras_model.layers:
#     layer.trainable = trainable
#     if layer.name == 'activation_40':
#         trainable = True
# compile(model.keras_model)
# model.keras_model.fit_generator(
#     train_generator,
#     initial_epoch=0,
#     epochs=10,
#     steps_per_epoch=config.STEPS_PER_EPOCH,
#     callbacks=callbacks,
#     validation_data=val_generator,
#     validation_steps=config.VALIDATION_STEPS,
# )

for layer in model.keras_model.layers:
    layer.trainable = True
compile(model.keras_model)
model.keras_model.fit_generator(
    train_generator,
    initial_epoch=0,
    epochs=200,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=config.VALIDATION_STEPS,
)
