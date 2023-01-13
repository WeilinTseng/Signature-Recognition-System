import cv2

# 圖片路徑
image_path = "C:/Weilin/Tensor/Signature Recognition System/scanpaper/Page_1.jpg"
image_path2 = "C:/Weilin/Tensor/Signature Recognition System/scanpaper/Page_2.jpg"
# 資料夾路徑
file_path_F = "C:/Weilin/Tensor/Signature Recognition System/OUTRF"
file_path_T = "C:/Weilin/Tensor/Signature Recognition System/OurT"

# 讀取圖片
img = cv2.imread(image_path)
img2 = cv2.imread(image_path2)

# 設定人
person = ['A', 'B', 'C', 'D', 'E', 'F']
person2 = ['G', 'H', 'I', 'J']

# 最左上角
X = 10
Y = 155

# 裁切區域的 x與 y 座標（左上角）
x = 10
y = 155

# 裁切區域的長度與寬度
w = 207
h = 91

i = 1

# *************  裁切圖片_Page1 ****************

# 擷取假資料
for k in person:
    # 從上到下擷取
    for i in range(1, 6):
        crop_img = img[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
        y = y + 93
        # print("陳永瑩_%s_%d.jpg" %(k, i))
        # 顯示圖片
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{file_path_F}/陳永瑩_%s_%d.jpg" % (k, i), crop_img)
    for j in range(1, 6):
        crop_img = img[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
        y = y + 93
        # print("曾威霖_%s_%d.jpg" %(k, j))
        # 顯示圖片
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{file_path_F}/曾威霖_%s_%d.jpg" % (k, j), crop_img)
    # 將擷取座標回復並跳到下一行
    x = x + 218
    y = Y

# 擷取真實資料
for l in range(1, 3):
    # 從上到下擷取
    for i in range(1, 11):
        crop_img = img[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
        y = y + 93
        # print("陳永瑩_%d.jpg" %(i))
        # 顯示圖片
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{file_path_T}/陳永瑩_%d.jpg" % i, crop_img)
    # 將擷取座標回復並跳到下一行
    x = x + 218
    y = Y

# *************  裁切圖片_Page2 ****************

x = X
y = Y

# 擷取假資料
for k in person2:
    # 從上到下擷取
    for i in range(1, 6):
        crop_img = img2[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
        y = y + 93
        # print("陳永瑩_%s_%d.jpg" %(k, i))
        # 顯示圖片
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{file_path_F}/陳永瑩_%s_%d.jpg" % (k, i), crop_img)
    for j in range(1, 6):
        crop_img = img2[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
        y = y + 93
        # print("曾威霖_%s_%d.jpg" %(k, j))
        # 顯示圖片
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{file_path_F}/曾威霖_%s_%d.jpg" % (k, j), crop_img)
    # 將擷取座標回復並跳到下一行
    x = x + 218
    y = Y

# 擷取真實資料
for l in range(1, 3):
    for i in range(1, 11):
        crop_img = img2[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
        y = y + 93
        # print("曾威霖_%d.jpg" %(i))
        # 顯示圖片
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{file_path_T}/曾威霖_%d.jpg" % (i), crop_img)
    # 將擷取座標回復並跳到下一行
    x = x + 218
    y = Y


# 擷取訪真假資料
for i in range(1, 6):
    crop_img = img2[y:y + h, x:x + w]
    crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
    y = y + 93
    # print("陳永瑩_曾威霖_%d.jpg" %(i))
    # 顯示圖片
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    cv2.imwrite(f"{file_path_F}/陳永瑩_曾威霖_%d.jpg" % (i), crop_img)
for j in range(1, 6):
    crop_img = img2[y:y + h, x:x + w]
    crop_img = cv2.resize(crop_img, (388, 184), interpolation=cv2.INTER_AREA)
    y = y + 93
    # print("曾威霖_陳永瑩_%d.jpg" %(j))
    # 顯示圖片
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    cv2.imwrite(f"{file_path_F}/曾威霖_陳永瑩_%d.jpg" % (j), crop_img)
