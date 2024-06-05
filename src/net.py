import tensorflow as tf
import keras
from keras import layers, optimizers, losses, metrics
import PIL
import numpy as np

### DATASET ###
TRAIN_DIR = r"cats_and_dogs_filtered\train"
VAL_DIR = r"cats_and_dogs_filtered\validation"
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)
validation_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)
class_names = train_ds.class_names
val_batches = tf.data.experimental.cardinality(validation_ds)
test_ds = validation_ds.take(val_batches // 5)
validation_ds = validation_ds.skip(val_batches // 5)
for images, _ in train_ds.take(1):
    IMG_SHAPE = images[0].shape


### PREPARATION PIPLINE ###
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


### AUGMENTATION ###
AUGMENT_LAYERS = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ]
)


### PREPROCESSING ###
PREPROCESS_MOBILE = keras.applications.mobilenet_v2.preprocess_input
RESCALE_LAYER = layers.Rescaling(1.0 / 127.5, offset=-1)


### BASE NET ###
FEATURE_EXTRACTION_LAYERS = keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet",
)
GLOBAL_AVG_LAYER = layers.GlobalAveragePooling2D()
PREDICTION_NEURON = layers.Dense(units=1)


### FREEZING ###
FEATURE_EXTRACTION_LAYERS.trainable = True
AT = 100
for layer in FEATURE_EXTRACTION_LAYERS.layers[:AT]:
    layer.trainable = False


### NET ARCHETECTURE ###
INPUTS = keras.Input(shape=IMG_SHAPE)
X = AUGMENT_LAYERS(INPUTS)
X = PREPROCESS_MOBILE(X)
X = FEATURE_EXTRACTION_LAYERS(X, training=False)
X = GLOBAL_AVG_LAYER(X)
X = layers.Dropout(0.2)(X)
OUTPUTS = PREDICTION_NEURON(X)
NET = keras.Model(INPUTS, OUTPUTS)

# FINE_TUNE_LRATE = 0.00001
# NET.compile(
#     loss=losses.BinaryCrossentropy(from_logits=True),
#     optimizer=optimizers.RMSprop(learning_rate=FINE_TUNE_LRATE),
#     metrics=[metrics.BinaryAccuracy(threshold=0.5, name="accuracy")],
# )

# FINE_TUNE_EPOCHS = 20
# HISTORY_FT = NET.fit(
#     train_ds,
#     epochs=FINE_TUNE_EPOCHS,
#     validation_data=validation_ds,
# )


### SAVE NET ###
# NET.save("NET.keras")