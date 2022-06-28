# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import time
from PIL import Image
import FeatureVector as FV

def selective_search(image, method="fast"):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if method == "fast":
        print("Iniciando metodo rapido")
        ss.switchToSelectiveSearchFast()
    else:
        print("Iniciando metodo calidad")
        ss.switchToSelectiveSearchQuality()
    t1 = time.time()
    rects = ss.process()
    t2 = time.time()
    print("Tiempo = ", t2-t1)
    return rects
    
model = ResNet50(weights="imagenet")
image = cv2.imread(r'.\Imagenes\IMG2I.jpg')


(H, W) = image.shape[:2]
print("H = ", H, " W = ", W)

AR = [4/3, 5/3, 5/4, 16/9, 16/10, 17/10]
ARc = [(4, 3), (5, 3), (5, 4), (16, 9), (16, 10), (17, 10)]
Div = max(H, W)/min(H, W)

for i in range(len(AR)):
    AR[i] = abs(AR[i]-Div)

ARval = np.argmin(AR)

H = int(np.floor(ARc[ARval][1]*(1024/ARc[ARval][0])))

image = cv2.resize(image, (1024, H))
(H, W) = image.shape[:2]
print("H = ", H, " W = ", W)

print("[INFO] performing selective search with fast method...")
rects = selective_search(image, method="fast")
print("[INFO] {} regions found by selective search".format(len(rects)))

resp = ["Cargador ", "Control remoto ", "Lana ", "Maletin ", "Mamadera ", "Mochila ", "Nada "]

m = FV.CargarRed()

images = []

for (x, y, w, h) in rects:
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue
        
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    r = FV.ProbarRed(m, roi)
    
    if r[1][r[0]]*100 > 1:
        text = resp[r[0]]+str(float(r[1][r[0]])*100)+"%"
        img = image[y:y + h, x:x + w]
        img = cv2.resize(img, (224, 224))
        images.append([text, img])

print("[INFO] {} filtred regions found by selective search".format(len(images)))

for i in images:
    cv2.imshow(i[0], i[1])
    cv2.waitKey(0)