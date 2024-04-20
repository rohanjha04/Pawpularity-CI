#import neccessary libraries
import tensorflow as tf
import numpy as np


#define the model
def load_model(path):
    #Define the model with resnet50 to extract features from images
    input_image = tf.keras.layers.Input(shape=(256, 256, 3))
    input_metadata = tf.keras.layers.Input(shape=(12,))
    base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=input_image,
                                                weights='imagenet')
    for layer in base_model.layers:
        layer.trainable= False
    x = base_model.output
    x2= tf.keras.layers.Flatten()(x)
    concat_layers = tf.keras.layers.Concatenate()([x2, input_metadata])
    l2= tf.keras.layers.Dense(512, activation='relu')(concat_layers)
    b1= tf.keras.layers.BatchNormalization()(l2)
    l3= tf.keras.layers.Dense(256, activation='relu')(b1)
    b2= tf.keras.layers.BatchNormalization()(l3)
    # l4= tf.keras.layers.Dense(128, activation='relu')(b2)
    # b3= tf.keras.layers.BatchNormalization()(l4)
    l5= tf.keras.layers.Dense(64, activation='relu')(b2)
    b4= tf.keras.layers.BatchNormalization()(l5)
    l6= tf.keras.layers.Dense(32, activation='relu')(b4)
    b5= tf.keras.layers.BatchNormalization()(l6)
    l7= tf.keras.layers.Dense(16, activation='relu')(b5)
    b6= tf.keras.layers.BatchNormalization()(l7)
    output_1 = tf.keras.layers.Dense(1, activation='sigmoid')(b6)
    model = tf.keras.models.Model(inputs=[input_image,input_metadata], outputs=output_1)
    
    model.load_weights(path)
    return model

#load the image
def load_image(image, image_size=(256, 256)):
    #Convert PIL image to tensor
    image = tf.keras.preprocessing.image.img_to_array(image)
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float16)
    image = image / 255.0 #normalize the image
    return image

#predict the image
def predict_image(image, metadata, model):
    # image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)
    prediction = model.predict([image, metadata])
    #round the prediction to nearest integer
    prediction = np.round(prediction*100)
    return prediction

def get_metadata(attribute_list):
    return np.array([1 if i else 0 for i in attribute_list]).reshape(1,12)

#Metadata columns are:
# ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']