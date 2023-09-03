'''
dir 图片路径
size 图片尺寸
'''
import os

import cv2
from PIL import Image
import numpy as np
import keras as ks
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self):
        self.benign_train = np.array(self.data_loader('./picture/benign', 224))
        self.malignant_train = np.array(self.data_loader('./picture/malignant', 224))
        self.malignant_test = np.array(self.data_loader('./picture/malignant', 224))
        self.benign_test = np.array(self.data_loader('./picture/benign', 224))

        # 创造标签用于标记图像 0矩阵表示良性 1矩阵表示恶性
        self.benign_train_label = np.zeros(len(self.benign_train))
        self.benign_test_label = np.zeros(len(self.benign_test))
        self.malignant_train_label = np.ones(len(self.malignant_train))
        self.malignant_test_label = np.ones(len(self.malignant_test))

        self.x_train = np.concatenate((self.benign_train , self.malignant_train))
        self.y_train = np.concatenate((self.benign_train_label , self.malignant_train_label))
        self.x_test = np.concatenate((self.benign_test , self.malignant_test))
        self.y_test = np.concatenate((self.benign_test_label , self.malignant_test_label))

        s = np.arange(self.x_train.shape[0])
        np.random.shuffle(s)
        # 随机打乱train
        self.x_train = self.x_train[s]
        self.y_train = self.y_train[s]

        s = np.arange(self.x_train.shape[0])
        np.random.shuffle(s)
        # 随机打乱test
        self.x_test = self.x_test[s]
        self.y_test = self.y_test[s]

        self.y_train = to_categorical(self.y_train , 2)
        self.y_test = to_categorical(self.y_test , 2)

        '''
            参数
            train_data => x_train
            train_target => y_ train
            test_size 样本占比：测试集占总体样本 测试集和训练集3/7分
            random_state 随机种子
        '''
        self.train_of_x , self.val_of_x , self.train_of_y , self.val_of_y = train_test_split(
            self.x_train , self.y_train,
            test_size=0.3,
            random_state=11
        )


    def data_loader(self, dir, size):
        IMG = []  # 图片
        # 打开该路径下的图像文件转换为RGB模式并返回numpy数组
        read = lambda i: np.asarray(Image.open(i).convert("RGB"))
        home_dir = sorted(os.listdir(dir))
        n = len(home_dir)
        for i in range(n):
            # 获取图片路径
            path = os.path.join(dir, home_dir[i])
            l = os.path.split(path)

            if "_mask" not in l[1]:
                # 正常png图片
                img = read(path)
                # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=562x471 at 0x1D3B7465150>
                # 需要resize缩放为224x224
                img = cv2.resize(img, (size, size))
                IMG.append(np.array(img))

        return IMG