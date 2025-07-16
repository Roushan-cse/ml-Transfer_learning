import tensorflow as tf
import os


def image_preprocessing(image):
    image = tf.stack([image]*3,axis=-1)
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize(image,[256,256])
    image = tf.expand_dims(image,axis=0)
    return image

def predictions(image):
    model = tf.keras.models.load_model(os.path.join("trained_model", "my_model.keras"))
    prediction = model.predict(image).argmax()
    return prediction
