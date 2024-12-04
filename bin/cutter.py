import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import shutil
from numpy.core.records import array
from numpy.core.shape_base import block
import time

class cutter:

    def __init__(self, img_path):

        self.img_origin = cv2.imread(img_path, 1)
        self.img = cv2.imread(img_path, 0)
        self.config_map = {
            "thresh": 200
        }
        

    @staticmethod
    def get_square_img(image):
        
        x, y, w, h = cv2.boundingRect(image)
        image = image[y:y+h, x:x+w]

        max_size = 18
        max_size_and_border = 24

        if w > max_size or h > max_size: # 有超过宽高的情况
            if w>=h: # 宽比高长，压缩宽
                times = max_size/w
                w = max_size
                h = int(h*times)
            else: # 高比宽长，压缩高
                times = max_size/h
                h = max_size
                w = int(w*times)
            # 保存图片大小
            image = cv2.resize(image, (w, h))


        xw = image.shape[0]
        xh = image.shape[1]

        xwLeftNum = int((max_size_and_border-xw)/2)
        xwRightNum = (max_size_and_border-xw) - xwLeftNum

        xhLeftNum = int((max_size_and_border-xh)/2)
        xhRightNum = (max_size_and_border-xh) - xhLeftNum
            
        img_large=np.pad(image,((xwLeftNum,xwRightNum),(xhLeftNum,xhRightNum)),'constant', constant_values=(0,0)) 
        
        return img_large

    @staticmethod
    def cut_img(img, mark_boxs, square_format=False):

        img_items = []
        for box in mark_boxs:
            img_org = img.copy()
            # 裁剪图片
            img_item = img_org[box[1]:box[3], box[0]:box[2]]
            
            if square_format: # 是否转化为方形
                img_item = cutter.get_square_img(img_item)

            img_items.append(img_item)

        return img_items

    @staticmethod
    def img_y_shadow(img_b):

        (height, width) = img_b.shape
        a = [0 for z in range(0, height)]
        for (i, row) in enumerate(img_b):
            for (j, dot) in enumerate(row):      
                if dot == 255:
                    a[i] += 1  
        return a


    @staticmethod
    def img_x_shadow(img_b):

        (height, width) = img_b.shape
        a = [0 for z in range(0, width)]
        for (i, row) in enumerate(img_b):
            for (j, dot) in enumerate(row):
                if dot == 255:
                    a[j] += 1
        return a


    @staticmethod
    def y_shadow_cut(a, box_in, width, height):

        left = box_in[0]
        top = box_in[1]
        inLine = False # 是否已经开始切分
        start = 0 # 某次切分的起始索引
        boxs_abs = list()
        boxs = list()
        for i in range(0, len(a)):
            if inLine == False and a[i] > 10:
                inLine = True
                start = i
            # 记录这次选中的区域[左，上，右，下]，上下就是图片，左右是start到当前
            elif inLine and i - start > 5 and a[i] < 10:
                inLine = False
                if i - start > 10:
                    t = max(start - 1, 0)
                    b = min(height, i + 1)
                    box = [0, t, width, b]
                    boxs.append(box)
                    box_abs = [left, top + t, left + width, top + b]
                    boxs_abs.append(box_abs)

        return boxs_abs, boxs
    
    @staticmethod
    def img_show_array(a):
        plt.imshow(a)
        plt.show()

    # 展示投影图， 输入参数arr是图片的二维数组，direction是x,y轴
    @staticmethod
    def show_shadow(arr, direction = 'x'):

        a_max = max(arr)
        if direction == 'x': # x轴方向的投影
            a_shadow = np.zeros((a_max, len(arr)), dtype=int)
            for i in range(0,len(arr)):
                if arr[i] == 0:
                    continue
                for j in range(0, arr[i]):
                    a_shadow[j][i] = 255
        elif direction == 'y': # y轴方向的投影
            a_shadow = np.zeros((len(arr),a_max), dtype=int)
            for i in range(0,len(arr)):
                if arr[i] == 0:
                    continue
                for j in range(0, arr[i]):
                    a_shadow[i][j] = 255

        cutter.img_show_array(a_shadow)  

    @staticmethod
    def y_cut(img, box, config_map):

        (height, width) = img.shape
        ret, img_b = cv2.threshold(img, config_map['thresh'], 255, cv2.THRESH_BINARY_INV)
        y_shadow_a = cutter.img_y_shadow(img_b)
        boxs_abs, boxs = cutter.y_shadow_cut(y_shadow_a, box, width, height)
        imgs = cutter.cut_img(img, boxs)

        return boxs_abs, imgs

    @staticmethod
    def x_shadow_cut(a, box_in, width, height):
        
        left = box_in[0]
        top = box_in[1]
        inLine = False # 是否已经开始切分
        start = 0 # 某次切分的起始索引
        boxs = list()
        boxs_abs = list()
        for i in range(0, len(a)):
            if inLine == False and a[i] > 2:
                inLine = True
                start = i
            # 记录这次选中的区域[左，上，右，下]，上下就是图片，左右是start到当前
            elif i - start > 5 and a[i] < 2 and inLine:
                inLine = False
                if i - start > 10:
                    l = max(start - 1, 0)
                    r = min(width, i + 1)
                    box = [l, 0, r, height]
                    boxs.append(box)
                    box_abs = [left + l, top, left + r, top + height]
                    boxs_abs.append(box_abs)
                    
        return boxs_abs, boxs


    # 保存图片
    @staticmethod
    def save_imgs(dir_name, imgs):
    
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name) 
        if not os.path.exists(dir_name):    
            os.makedirs(dir_name)

        img_paths = []
        for i, img in enumerate(imgs):
            file_path = f'{dir_name}/part_{i}.jpg'
            cv2.imwrite(file_path, img)
            img_paths.append(file_path)
        
        return img_paths

    @staticmethod
    def x_cut_with_delate(img, box, config_map):

        (height, width) = img.shape
        ret, img_b = cv2.threshold(img, config_map['thresh'], 255, cv2.THRESH_BINARY_INV)
        
        # delate 6 times
        kernel = np.ones((3,3), np.uint8)
        img_b_d = cv2.dilate(img_b, kernel, iterations=6)

        # calculate x shadow
        x_shadow_a = cutter.img_x_shadow(img_b_d)
        boxs_abs, boxs = cutter.x_shadow_cut(x_shadow_a, box, width, height)
        imgs = cutter.cut_img(img, boxs)

        return boxs_abs, imgs
    

    @staticmethod
    def x_cut(box, img, config_map):

        (height, width) = img.shape
        ret, img_b = cv2.threshold(img, config_map['thresh'], 255, cv2.THRESH_BINARY_INV)
        
        x_shadow_a = cutter.img_x_shadow(img_b)
        print(x_shadow_a)
        boxs_abs, boxs = cutter.x_shadow_cut(x_shadow_a, box, width, height)
        print(boxs)
        #cutter.show_shadow(x_shadow_a)
        imgs = cutter.cut_img(img_b, boxs, True)

        return boxs, imgs


    def divImg(self):

        # img to row
        (width, height) = self.img.shape
        box = [0, 0, width, height]
        row_boxs, row_imgs = cutter.y_cut(self.img, box, self.config_map)

        self.boxs_list = list()
        self.imgs_list = list()
        for (i, (row_img, row_box)) in enumerate(zip(row_imgs, row_boxs)):
            # row into blocks
            block_boxs, block_imgs = cutter.x_cut_with_delate(row_img, row_box, self.config_map)
            cutter.save_imgs(f"/Users/taoliu/Documents/git/github/img2txt/tmp/{i}", block_imgs)

            row_boxs_list = list()
            row_imgs_list = list()
            for (j, (block_box, block_img)) in enumerate(zip(block_boxs, block_imgs)):
                # block to chars
                #print(f"block_box {block_box}")
                char_boxs_list, char_imgs_list = cutter.x_cut(block_box, block_img, self.config_map)
                cutter.save_imgs(f"/Users/taoliu/Documents/git/github/img2txt/tmp/{i}/{j}", char_imgs_list)
                row_boxs_list.append(char_boxs_list)
                row_imgs_list.append(char_imgs_list)

            self.boxs_list.append(row_boxs_list)
            self.imgs_list.append(row_imgs_list)


if __name__ == '__main__':
    # read image
    img_path = '/Users/taoliu/Documents/git/github/img2txt/image/question.png'
    exp1 = cutter(img_path)

    exp1.divImg()

    #from char_model import char_model
    import cnn
    model = cnn.create_model()
    model.load_weights('/Users/taoliu/Documents/git/github/img2txt/checkpoint/char_checkpoint.weights.h5')
    class_name = np.load('/Users/taoliu/Documents/git/github/img2txt/checkpoint/class_name.npy')

    # 遍历行
    for (i, row_img_list) in enumerate(exp1.imgs_list):
        # 遍历块
        for (j, block_img_list) in enumerate(row_img_list):
            char_imgs_np = np.array(block_img_list)
            results = cnn.predict(model, char_imgs_np, class_name)
            print('recognize result:', results)