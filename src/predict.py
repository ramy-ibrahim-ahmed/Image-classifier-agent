import keras
import tensorflow as tf
from PIL import Image


class_names = ["Cats", "Dogs"]
model = keras.models.load_model("src/NET.keras")
# cat = r"cats_and_dogs_filtered\validation\cats\cat.2008.jpg"


def predict_(img):
    img = img.resize((160, 160))
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    prediction = (model.predict(img)).flatten()
    propability = tf.nn.sigmoid(prediction)
    prediction = tf.where(propability < 0.5, 0, 1)
    return class_names[prediction.numpy()[0]]