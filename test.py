
import os
"""
data = os.path.join(os.getcwd(), 'ImageSets/Segmentation/nimeide.jpg')
print(data)
fname = data.split('/')[-1].split('.')[0].encode('utf8')
print(fname)
"""

#txt_fname = 'D:\\Documents\\AI\\ThirdParty\\models\research\\data\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt'
txt_fname = 'D:\\Documents\\AI\\ThirdParty\models\\research\\data\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt'
print(txt_fname)
with open(txt_fname, 'r') as f:
        images = f.read().split()