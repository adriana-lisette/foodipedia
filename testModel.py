# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:31:43 2022

@author: Adriana Garcia
"""

from tensorflow.keras.models import model_from_json
import tensorflow as tf

from keras_preprocessing import image
import numpy as np

#get model from json
json_file = open('model2.json', 'r')
loaded_model = json_file.read()
json_file.close()
NewModel = model_from_json(loaded_model)

# asign weights 
NewModel.load_weights("model.h5")
print("Loaded model from disk")

class_names = {0:'chile en nogada', 1:'churros', 2:'enchiladas', 3:'guacamole', 4:'tacos'}

#compile new model
NewModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def getDishName(foodImg):   
    img = image.load_img(foodImg, target_size=(150,150))
    x = image.img_to_array(img)
    x = tf.expand_dims(x, 0)
    
    predictions = NewModel.predict(x)
    score = tf.nn.softmax(predictions[0])
    
    return class_names[np.argmax(score)]

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
