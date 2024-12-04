from __future__ import print_function
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import shutil
import time

# %% 要生成的文本
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}
#label_dict = {0: '×'}

# 文本对应的文件夹，给每一个分类建一个文件
for value,char in label_dict.items():
    train_images_dir = "dataset"+"/"+str(value)
    if os.path.isdir(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir)

def getsize(font, char):
    left, top, right, bottom = font.getbbox(char)
    width = right - left
    height = bottom
    return width, height

# %% 生成图片
def makeImage(label_dict, font_path, font_index, width=24, height=24, rotate = 0):

    # 从字典中取出键值对
    for value,char in label_dict.items():
        # 创建一个黑色背景的图片，大小是24*24
        img = Image.new("RGB", (width, height), "black") 
        draw = ImageDraw.Draw(img)
        # 加载一种字体,字体大小是图片宽度的90%
        font = ImageFont.truetype(font_path, int(width*0.9))
        # 获取字体的宽高
        font_width, font_height = getsize(font, char)
        # 计算字体绘制的x,y坐标，主要是让文字画在图标中心
        x = (width - font_width - font.getbbox(char)[0]) / 2
        y = (height - font_height-font.getbbox(char)[1]) / 2
        # 绘制图片，在那里画，画啥，什么颜色，什么字体
        draw.text((x,y), char, (255, 255, 255), font)
        # 设置图片倾斜角度
        img = img.rotate(rotate)
        # 命名文件保存，命名规则：dataset/编号/img-编号_r-选择角度_时间戳.png
        #time_value = int(round(time.time() * 1000))
        img_path = f"/Users/taoliu/Documents/git/github/img2txt/dataset/{value}/img_{value}_{font_index}_{rotate+10}.png"
        #img_path = f"/Users/taoliu/Documents/git/github/img2txt/dataset/{value}/{font_name}.png"
        img.save(img_path)
        
# %% 存放字体的路径
font_dir = "/Users/taoliu/Documents/git/github/img2txt/fonts"
font_index = 0
for font_name in os.listdir(font_dir):
    font_index += 1
    # 把每种字体都取出来，每种字体都生成一批图片
    path_font_file = os.path.join(font_dir, font_name)
    # 倾斜角度从-10到10度，每个角度都生成一批图片
    for k in range(-10, 10, 1): 
    #for k in range(0, 1, 1): 
        # 每个字符都生成图片
        makeImage(label_dict, path_font_file, font_index, rotate = k)
