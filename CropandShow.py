import pyautogui
import cv2
import numpy as np
import tkinter as tk
from tkinter.constants import CENTER
from PIL import Image, ImageTk

# 指定檔案儲存位置
saveFile = "C:/Weilin/Tensor/Signature Recognition System/saveFile"


# 擷取指令
def plus():

    # 擷取螢幕畫面
    screenshot = pyautogui.screenshot()

    # 將螢幕截圖轉換為NumPY陣列
    image = np.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 將擷取區域(ROI)定義為方形
    roi = cv2.selectROI('roi', image, False, False)

    # 擷取區域
    cropped_image = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    # 將擷取後的圖像儲存到文件夾中
    cv2.imwrite(r"saveFile/cropped_image.jpg", cropped_image)

    # 關閉擷取螢幕畫面
    cv2.destroyAllWindows()


# 擷取簽名主程式
def Crop_Screen():
    # 建立主視窗 win
    win = tk.Tk()

    # 設定視窗標題
    win.title('GUI')

    # 設定視窗大小為 300x200，視窗（左上角）在螢幕上的座標位置為 (550, 400)
    win.geometry('300x200+550+400')
    win.resizable(False, False)

    # 設定擷取螢幕按鍵
    Button1 = tk.Button(win, text="Crop", font=('Arial', 30), height=1, width=11, command=plus)
    Button1.place(x=150, y=60, anchor=CENTER)

    # 設定退出螢幕按鍵
    Button2 = tk.Button(win,text='Click and Quit', font=('Arial', 30), height=1, width=11, command=win.destroy)
    Button2.place(x=150, y=140, anchor=CENTER)

    win.mainloop()


# 顯示簽名辨識結果
def show(Val, testImage_path):
    # 建立主視窗 window
    window = tk.Tk()

    # 設定視窗標題
    window.title('Result')

    # 設定視窗大小為 400x200，視窗（左上角）在螢幕上的座標位置為 (550, 400)
    window.geometry("400x200+550+400")
    window.resizable(False, False)

    # 標示文字
    img = Image.open(testImage_path)
    img = img.resize((200, 100), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)

    # 設定標籤來顯示文字或圖像
    label = tk.Label(window, image=img)
    label.image = img
    label.place(x=200, y=75, anchor=CENTER)

    # 顯示簽名真假
    if Val == [1]:
        label1 = tk.Label(window, text='True', bg='#4CBB17', font=('Arial', 20), width=5, height=1)
        label1.place(x=200, y=160, anchor=CENTER)
    else:
        label1 = tk.Label(window, text='False', bg='#C70039', font=('Arial', 20), width=5, height=1)
        label1.place(x=200, y=160, anchor=CENTER)

    window.mainloop()