import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 指定亂數種子
seed = 7
np.random.seed(seed)

# 載入真實簽名資料庫
gen = [glob(r"C:/Weilin/Tensor/Signature Recognition System/Offline Genuine/*.*"),
       glob(r"C:/Weilin/Tensor/Signature Recognition System/gen/*.*"),
       glob(r"C:/Weilin/Tensor/Signature Recognition System/OT/*.*")]

# 載入仿冒簽名資料庫
forg = [glob(r"C:/Weilin/Tensor/Signature Recognition System/Offline Forgeries/*.*"),
        glob(r"C:/Weilin/Tensor/Signature Recognition System/forg/*.*"),
        glob(r"C:/Weilin/Tensor/Signature Recognition System/OF/*.*")]


# 定義以下為array陣列
X = []
Y = []
X_train = []
Y_train = []
X_test = []
Y_test = []


# laplacian預處理
def composite_laplacian(f):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    temp = cv2.filter2D(f, cv2.CV_32F, kernel)
    g = np.uint8(np.clip(temp, 0, 255))
    return g


# 輸入真實圖片
for data in range(len(gen)):
    for i in gen[data]:
        image = cv2.imread(i)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.medianBlur(image, 5)
        print(image.shape)
        # image = cv2.bilateralFilter(image, 15, 75, 75)
        # image = composite_laplacian(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.threshold(image, 127, 255, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 64))
        X.append(image)
        Y.append(0)

# 輸入仿冒圖片
for data in range(len(forg)):
    for j in forg[data]:
        image = cv2.imread(j)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.medianBlur(image, 5)
        # image = cv2.bilateralFilter(image, 15, 75, 75)
        # image = composite_laplacian(image)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret, frame = cv2.threshold(image, 127, 255, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 64))
        X.append(image)
        Y.append(1)

# 轉換成array陣列過後進行打亂
X = np.array(X)
Y = np.array(Y)
X, Y = shuffle(X, Y)
print(X.shape)
print(Y.shape)

# 分割資料集
X_train, Y_train = X[:8500], Y[:8500]
X_test, Y_test = X[8500
                   :], Y[8500:]
# 正規化
X_train = X_train / 255
X_test = X_test / 255
# One-hot編碼
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

'''
train_datagen = ImageDataGenerator(
    width_shift_range=0.01,
    height_shift_range=0.01,
    zoom_range=0.01,
)
train_generator = train_datagen.flow(
             X_train, Y_train,
             batch_size=10)
'''

# 定義模型
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), padding="same",
                 input_shape=(64, 128, 3), activation="relu"))

# ,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01)
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(3, 3), padding="same",
                 activation="relu"))
# ,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l2(0.01))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
# model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(2, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001),
              metrics=["accuracy"])

# 訓練模型

'''
history = model.fit_generator(
          train_generator,
          steps_per_epoch=int(8500/10),
          epochs=40, verbose=2,
          validation_data=(X_test, Y_test))
# 建立 EarlyStopping 物件
es = EarlyStopping(monitor="val_accuracy", mode="min",
                   verbose=1, patience=7)
history = model.fit(X_train, Y_train, validation_split=0.2,
          epochs=40, batch_size=128,
          verbose=2, callbacks=[es])
'''
history = model.fit(X_train, Y_train, validation_split=0.2,
                    epochs=30, batch_size=128, verbose=2)

# 評估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
val_loss, val_accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = {:.2f}".format(val_accuracy))

# 儲存Keras模型
print("Saving Model: mnist.h5 ...")
model.save("signatureRecognitionNew.h5")

# 顯示訓練和驗證準確率
accuracy = history.history["accuracy"]
epochs = range(1, len(accuracy) + 1)
val_accuracy = history.history["val_accuracy"]
plt.plot(epochs, accuracy, "bo-", label="Training Acc")
plt.plot(epochs, val_accuracy, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()