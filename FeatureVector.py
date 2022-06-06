# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import cv2

# For measuring the inference time.
import time

import pylab

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import joblib  
import os

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)
  pylab.show()
  
def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def GraficasPrediccion(historial):
    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']

    loss = historial.history['loss']
    val_loss = historial.history['val_loss']

    rango_epocas = range(50)

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(rango_epocas, acc, label='Precision entrenamiento')
    plt.plot(rango_epocas, val_acc, label='Precision pruebas')
    plt.legend(loc='lower right')
    plt.title('Precision de entrenamiento y pruebas')

    plt.subplot(1,2,2)
    plt.plot(rango_epocas, loss, label='Perdida de entrenamiento')
    plt.plot(rango_epocas, val_loss, label='Perdida de pruebas')
    plt.legend(loc='upper right')
    plt.title('Perdida de entrenamiento y pruebas')
    plt.show()

'''
datagen = ImageDataGenerator( rescale=1. / 255,
                              rotation_range = 30,
                              width_shift_range = 0.25,
                              height_shift_range = 0.25,
                              shear_range = 15,
                              zoom_range = [0.5, 1.5],
                              validation_split = 0.2)

data_gen_entrenamiento = datagen.flow_from_directory('.\dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory('.\dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='validation')

for imagen, etiqueta in data_gen_entrenamiento:
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i])
    break
plt.show()
'''
#downloaded_image_path = get_and_resize_image(".\Imagenes\IMG2I.jpg" , 1280, 856, True)

module_handle = "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5"

m = tf.keras.Sequential([
    hub.KerasLayer(module_handle,
                   trainable=False),
    tf.keras.layers.Dense(6, activation='softmax')
])
m.build([None, 224, 224, 3])

m.summary()

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

m.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

checkpoint_path = "training_1/Model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
m.load_weights(checkpoint_path)

#path = get_and_resize_image(".\Imagenes\\5(3).jpg" , 224, 224, True)


for i in range(7):
    img = Image.open(".\Imagenes\\"+str(i+1)+".jpg")
    img = np.array(img).astype(float)/255

    img = cv2.resize(img, (224, 224))

    result = m(img.reshape(-1, 224, 224, 3))

    print(np.argmax(result[0], axis = -1))


#print(data_gen_pruebas[0])

# Re-evaluate the model
#loss, acc = m.evaluate( data_gen_pruebas[0], 
#                        data_gen_pruebas[1],
#                        verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Create a callback that saves the model's weights
'''
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

EPOCAS = 50

h = m.fit(
    data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas,
    callbacks=[cp_callback])
    
joblib.dump(m, 'modelo_entrenado.pkl')

GraficasPrediccion(h)
'''