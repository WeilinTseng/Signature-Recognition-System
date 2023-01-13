import cv2
import numpy as np

# 指定檔案儲存位置
saveFile = "C:/Weilin/Tensor/Signature Recognition System/saveFile"


# 顯示圖片
def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 調整圖片大小
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 輪廓檢測
def edge_detection(img_path):
    # *********  預處理 ****************
    # 讀取輸入
    img = cv2.imread(img_path)
    # 座標也會相同變換
    ratio = img.shape[0] / 500.0
    orig = img.copy()

    image = resize(orig, height=500)
    # 預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    # *************  輪廓檢測 ****************
    # 輪廓檢測
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # 遍歷輪廓
    for c in cnts:
        # 計算輪廓近似
        peri = cv2.arcLength(c, True)
        # c表示輸入的點集，epsilon表示從原始輪廓到近似輪廓的最大距離，它是一個準確度引數
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4個點的時候就拿出來
        if len(approx) == 4:
            screenCnt = approx
            break

    # res = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

    return orig, ratio, screenCnt

    # show(res)


# 邊緣偵測
def order_points(pts):
    # 一共四個座標點
    rect = np.zeros((4, 2), dtype='float32')

    # 按順序找到對應的座標0123 分別是左上，右上，右下，左下
    # 計算左上，由下
    # numpy.argmax(array, axis) 用於返回一個numpy陣列中最大值的索引值
    s = pts.sum(axis=1)  # [2815.2   1224.    2555.712 3902.112]
    print(s)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 計算右上和左
    # np.diff()  沿著指定軸計算第N維的離散差值  後者-前者
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透視轉換
def four_point_transform(image, pts):
    # 獲取輸入座標點
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 計算輸入的w和h的值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 變化後對應座標位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype='float32')

    # 計算變換矩陣
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回變換後的結果
    return warped


# 對透視變換結果進行處理
def get_image_processingResult():
    img_path = 'saveFile/cropped_image.jpg'
    orig, ratio, screenCnt = edge_detection(img_path)
    # screenCnt 為四個頂點的座標值，但是我們這裡需要將影像還原，即乘以以前的比率
    # 透視變換  這裡我們需要將變換後的點還原到原始座標裡面
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # 二值處理
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    cv2.imwrite('saveFile/scan.jpg', dilation)
    thresh_resize = resize(dilation, height=400)
