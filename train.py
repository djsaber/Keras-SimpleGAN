# coding=gbk
from model import SimpleGAN
from utils import load_mnist

#---------------------------------设置参数-------------------------------------
latent_shape = (100,)       # 噪声维度
output_shape = (28,28)    # 样本维度
batch_size = 128            # 训练批次大小
epochs = 30000              # 训练轮数
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
data_path = "D:/科研/python代码/炼丹手册/GAN/datasets/fashion-mnist/"
save_path = "D:/科研/python代码/炼丹手册/GAN/save_models/"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
trainX, _ = load_mnist(data_path, kind='train')
trainX = trainX.reshape(((len(trainX),)+output_shape))
trainX = trainX / 255
print('trainX shape:', trainX.shape)
#-----------------------------------------------------------------------------


#---------------------------------搭建模型-------------------------------------
sgan = SimpleGAN(
    latent_shape=latent_shape,
    output_shape=output_shape
    )
#-----------------------------------------------------------------------------


#--------------------------------训练和保存-------------------------------------
sgan.fit(
    x = trainX,
    epochs=epochs,
    batch_size=batch_size
    )
sgan.save_weight(save_path)
#-----------------------------------------------------------------------------