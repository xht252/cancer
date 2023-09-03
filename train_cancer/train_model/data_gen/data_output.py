from keras.preprocessing.image import ImageDataGenerator

class gen_data:
    def __init__(self):
        # batch 表示训练样本数
        # 这里推测20一组为好
        # 过大会过拟合
        self.batch = 20
        # keras 提供的数据生成器
        '''
        zoom_range 随机缩放的幅度
        rotation_range 数据提升时图片随机转动的角度
        horizontal_flip 图片随机水平翻转
        vertical_flip 图片竖直翻转
        '''
        self.tr_gen = ImageDataGenerator(
            zoom_range=2,
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True
        )