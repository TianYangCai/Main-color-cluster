import math
from PIL import Image, ImageFile
import os
import numpy as np
from core import ColorPalette
ImageFile.LOAD_TRUNCATED_IMAGES = True
import colorsys
import cv2



M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])


# im_channel取值范围：[0,1]
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion



def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB = np.array([gamma(c) for c in rgb])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab


# endregion

# region Lab 转 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb





def changeImg_rgb2lab (img):
    w = img.shape[0]
    h = img.shape[1]
    lab = np.zeros((w,h,3))
    for i in range(w):
        for j in range(h):
            Lab = RGB2Lab(img[i,j])
            lab[i, j] = (Lab[0], Lab[1], Lab[2])

    return lab










#计算普通RGB图片的色彩聚类
def SingleImgCal(file_name):
    img = Image.open(file_name).convert("RGB")
    img = np.array(img)

    pixels = []
    for col in range(img.shape[0]):
        for row in range(img.shape[1]):
            if img[col][row][0] > 50 and img[col][row][1] > 50 and img[col][row][2] > 50:
                pixels.append(img[col][row])

    #print(img.shape)
    #print(np.array(pixels))
    pal = ColorPalette(np.array(pixels), show_clustering = False)
    topColor = pal.get_top_colors(n = 8, ratio = True, rounded = True)
    topColorRGB = []
    topColorWeight = []
    for i in range(0, len(topColor)):
        topColorRGB.append([topColor[i][0][0], topColor[i][0][1], topColor[i][0][2]])
        topColorWeight.append(topColor[i][1])


    # 绘制最靠近壁纸的topColor的色卡图
    im = Image.new("RGB", (800, 1600))
    print(topColorRGB)
    print(topColorWeight)
    for index in range(len(topColorRGB)):
        im.paste(tuple(topColorRGB[index]), (0, (200 * index), 800, (200 * (index + 1))))
    im.save("images/result_RGB.jpg")

    topColorRGB_result = []
    topColorWeight_resutlt = []
    for i in range(len(topColorWeight)):
        if topColorWeight[i] > 5:
            topColorRGB_result.append(topColorRGB[i])
            topColorWeight_resutlt.append(topColorWeight[i])

    return topColorRGB_result, topColorWeight_resutlt






#计算普通LAB图片的色彩聚类
def SingleImgCal_LAB(file_name):
    #img = Image.open(file_name).convert("RGB")
    img = cv2.imread(file_name)
    img = changeImg_rgb2lab(img)
    #img = np.array(img)

    pixels = []
    for col in range(img.shape[0]):
        for row in range(img.shape[1]):
            if img[col][row][0] != 0 and img[col][row][1] != 0 and img[col][row][2] != 0:
                pixels.append(img[col][row])


    pal = ColorPalette(np.array(pixels), show_clustering = False)
    topColor = pal.get_top_colors(n = 6, ratio = True, rounded = True)
    topColorLAB = []
    topColorWeight = []
    for i in range(0, len(topColor)):
        topColorLAB.append(Lab2RGB(topColor[i][0]))
        topColorWeight.append(topColor[i][1])


    # 绘制最靠近壁纸的topColor的色卡图
    im = Image.new("RGB", (800, 1600))
    print(topColorLAB)
    print(topColorWeight)
    for index in range(len(topColorLAB)):
        im.paste(tuple(topColorLAB[index]), (0, (200 * index), 800, (200 * (index + 1))))
    im.save("images/result_LAB.jpg")

    topColorLAB_result = []
    topColorWeight_resutlt = []
    for i in range(len(topColorWeight)):
        if topColorWeight[i] > 5:
            topColorLAB_result.append(topColorLAB[i])
            topColorWeight_resutlt.append(topColorWeight[i])

    return topColorLAB_result, topColorWeight_resutlt










#计算灰度图亮度聚类
def SingleImgCal_gray(file_name):
    img = Image.open(file_name).convert('L')

    img = np.array(img)

    pixels = []
    for col in range(img.shape[0]):
        for row in range(img.shape[1]):
            if img[col][row] != 0:
                pixels.append(img[col][row])


    pal = ColorPalette(np.array(pixels), show_clustering = False)
    topColor = pal.get_top_colors(n = 8, ratio = True, rounded = True)
    topColorGray = []
    topColorWeight = []
    for i in range(0, len(topColor)):
        topColorGray.append(topColor[i][0])
        topColorWeight.append(topColor[i][1])


    # 绘制最靠近壁纸的topColor的色卡图
    im = Image.new("RGB", (800, 1600))
    print(topColorGray)
    print(topColorWeight)
    for index in range(len(topColorGray)):
        if isinstance(topColorGray[index], int):
            im.paste(topColorGray[index], (0, (200 * index), 800, (200 * (index + 1))))
        else:
            im.paste(tuple(topColorGray[index]), (0, (200 * index), 800, (200 * (index + 1))))
    im = im.convert("L")
    im.save("images/gray_result.jpg")

    topColorRGB_result = []
    topColorWeight_resutlt = []
    for i in range(len(topColorWeight)):
        if topColorWeight[i] > 5:
            topColorRGB_result.append(topColorGray[i])
            topColorWeight_resutlt.append(topColorWeight[i])

    return topColorRGB_result, topColorWeight_resutlt


#计算两个颜色类别的加权距离
def calDistanceWeight(frontColor,frontWeight,backColor,backWeight):

    result = np.sqrt(sum((np.array(frontColor)*frontWeight - np.array(backColor)*backWeight) ** 2))
    return result


#计算前后景颜色差异距离
def distanceFrontBack(frontColor,frontWeight,backColor,backWeight):
    result = []
    for i in range(len(frontColor)):
        for j in range(len(backColor)):
            result.append(calDistanceWeight(frontColor[i],backColor[j],frontWeight[i]/100,backWeight[j]/100))

    return np.mean(result)



#图像色调标准差
def stdColorLAB(topColorLAB):
    return np.std(topColorLAB)







if __name__ == "__main__":
    topColorRGB, topColorWeight = SingleImgCal("images/demo5.jpg")
    topColorLAB, topColorWeight_LAB = SingleImgCal_LAB("images/demo5.jpg")
    #topColorGray, topColorWeight_Gray = SingleImgCal_gray("images/demo5.jpg")
    #print(distanceFrontBack(topColorGray, topColorWeight_Gray, topColorGray, topColorWeight_Gray))

    #print(stdColorLAB(topColorLAB))
