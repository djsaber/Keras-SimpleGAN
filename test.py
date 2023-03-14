# coding=gbk
from model import SimpleGAN


#---------------------------------���ò���-------------------------------------
latent_shape = (100,)        # ����ά��
output_shape = (28,28)       # ����ά��
batch_size = 7               # ������������
#-----------------------------------------------------------------------------


#---------------------------------����·��------------------------------------
load_path = "D:/����/python����/�����ֲ�/GAN/save_models/"
save_path = "D:/����/python����/�����ֲ�/GAN/images/"
#-----------------------------------------------------------------------------


#---------------------------------����ģ��-------------------------------------
sgan = SimpleGAN(
    latent_shape=latent_shape,
    output_shape=output_shape
    )
sgan.load_weight(load_path)
#-----------------------------------------------------------------------------


#---------------------------------��������-------------------------------------
sgan.sample_imgs(batch_size, save_path=save_path, cmap="gray")
#-----------------------------------------------------------------------------