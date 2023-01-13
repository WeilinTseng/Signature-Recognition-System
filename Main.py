import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from PIL import Image, ImageTk

import CropandShow

import ScanSig

# 指定亂數種子
seed = 7
np.random.seed(seed)

# 擷取待測物資料集
CropandShow.Crop_Screen()

# 輸出待測物資料集
ScanSig.get_image_processingResult()

# 載入待測物資料集
testImage_path = r"C:/Weilin/Tensor/Signature Recognition System/saveFile/scan.jpg"


# 待測物所需之預處理
def test_process(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret, frame = cv2.threshold(image, 127, 255, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(frame, (64, 128))
    return image


# 進行待測物所需之預處理
testImage = test_process(testImage_path)
# 調整待測物的大小
testImage = testImage.reshape((1, 64, 128, 3))


# 建立Keras的Sequential模型
model = Sequential()
model = load_model("signatureRecognition.h5")
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# 計算分類的預測值
print("\nPredicting ...")
Y_pred = model.predict(testImage)
classes_x = np.argmax(Y_pred, axis=1)
print(Y_pred)
print(classes_x)

# 輸出待測物的分析結果
CropandShow.show(classes_x, testImage_path)
