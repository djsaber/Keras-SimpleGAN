# Keras-SimpleGAN
基于Keras搭建一个简单的GAN，用fashion-mnist数据集对GAN进行训练，完成模型的保存和加载以及测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：将数据集文件解压至此<br />
2. /save_models：保存训练好的模型权重文件，包括生成器权重和判别器权重两个文件<br />
3. /images：保存生成的样本<br /><br />

GAN概述<br />
生成对抗网络是非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。<br />
该方法由Google研究人员Ian Goodfellow于2014年提出。生成对抗网络由一个生成网络与一个判别网络组成。<br />
1. 生成网络从潜在空间（latent space）中随机采样作为输入，其输出结果需要尽量模仿训练集中的真实样本。<br />
2. 判别网络的输入则为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能分辨出来，而生成网络则要尽可能地欺骗判别网络。<br />
两个网络相互对抗、不断调整参数，最终目的是使判别网络无法判断生成网络的输出结果是否真实。<br /><br />

数据集：<br />
fashion-mnist：包含十个类别的服装衣物数据集,分别是：<br />
t-shirt（T恤），trouser（牛仔裤），pullover（套衫），dress（裙子），coat（外套），sandal（凉鞋），shirt（衬衫），sneaker（运动鞋），bag（包），ankle boot（短靴）。<br />
训练集/测试集包含60000/1000张灰度图<br />
链接：https://pan.baidu.com/s/1iBNmixiuORjpHjDSFephcg?pwd=52dl 提取码：52dl<br /><br />

生成器输出层采用sigmoid激活函数，所以将图片像素值缩放到0-1之间<br />
若使用tanh激活函数，则需要将图片像素值缩放到-1-1之间<br />

通过堆叠简单的全连接层Keras.layers.Dense来构建gan的生成器和判别器<br />
生成器中使用了BN标准化层，判别器中未使用，均使用adam优化器更新参数<br />
每个eopch中的训练过程如下：<br />
1. 随机采样batch size个真实样本<br />
2. 随机采样batch size个随机噪声，输入生成器获得相等数量的生成样本<br />
3. 将真实样本和生成样本输入判别器进行训练，真实样本标签为1，生成样本标签为0<br />
4. 将生成器和判别器连接起来，并冻结判别器参数，使其不参与更新<br />
5. 随机采样batch size个随机噪声，输入连接起来的模型进行训练，标签为1<br /><br />

采用1:1训练比，即每个epoch判别器和生成器各训练一次，根据不同场景可以更改此项。<br />
需要注意的是，Keras训练过程中模型的状态依据的是model.compile()时的状态。<br />
在定义判别器后，通过compile()编译判别器，然后连接生成器和判别器构成combined模型，并将判别器的trainable设置为False，再通过compile()编译combined模型。<br />
在训练时判别器和combined模型各自保持编译时的状态，所以在训练过程中不需要反复将判别的的trainable参数设置为True和False<br />



