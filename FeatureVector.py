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

# For measuring the inference time.
import time

import pylab

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_and_resize_image(dr, new_width=256, new_height=256, display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  pil_image = Image.open(dr)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename

datagen = ImageDataGenerator( rescale=1. / 255,
                              rotation_range = 30,
                              width_shift_range = 0.25,
                              height_shift_range = 0.25,
                              shear_range = 15,
                              zoom_range = [0.5, 1.5],
                              validation_split = 0.2)

data_gen_entrenamiento = datagen.flow_from_directory('.\dataset', target_size=(640,640),
                                                     batch_size=32, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory('.\dataset', target_size=(640,640),
                                                     batch_size=32, shuffle=True, subset='validation')

for imagen, etiqueta in data_gen_entrenamiento:
  for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i])
  break
plt.show()

def GraficasPrecicion(historial):
    acc = historial.history['accurancy']
    val_acc = historial.history['val_accurancy']

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

#downloaded_image_path = get_and_resize_image(".\Imagenes\IMG2I.jpg" , 1280, 856, True)

module_handle = "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5"
'''
detector = hub.load(module_handle)
run_detector(detector, downloaded_image_path)'''


m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5",
                   trainable=False),
    tf.keras.layers.Dense(8, activation='softmax')
])
m.build([None, 224, 224, 3])

m.summary()

m.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accurancy'])

EPOCAS = 50

h = m.fit(
    data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
    validation_data=data_gen_pruebas)

GraficasPrediccion(h)
