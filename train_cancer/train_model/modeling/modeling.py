from keras.models import Sequential
from keras import layers

class breast_train_test:
    def __init__(self):
        # 构造模型序列
        self.model = Sequential()
    '''
    resnet：DenseNet201网络
    study_rate 学习率
    '''
    def build(self , resnet , study_rate):
        self.model.add(resnet)
        # GlobalAveragePooling2D每个通道值各自加起来再求平均,只剩下个数与平均值两个维度
        self.model.add(layers.GlobalAveragePooling2D())
        # dropout 减少中间神经元个数 保留概率为0.5
        self.model.add(layers.Dropout(0.5))
        # BatchNormalization 每一个批次的数据中标准化前一层的激活项
        self.model.add(layers.BatchNormalization())
        # dense 全连接层 输出维度为2 activation激活函数为softmax在思路整理中给出
        self.model.add(layers.Dense(2, activation='softmax'))