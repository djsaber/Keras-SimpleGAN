# coding=gbk

from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.layers import Input, Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt


class Generator(Model):
    '''生成器，从随机噪声中生成样本'''
    def __init__(
        self,
        input_shape,
        output_shape,
        **kwargs):
        super().__init__(**kwargs)
        self.flatten = Flatten()
        self.blocks = [
            [Dense(256),
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
            [Dense(512),
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
            [Dense(1024),
             BatchNormalization(momentum=0.8),
             LeakyReLU(alpha=0.2)],
             [Dense(K.prod(output_shape), activation='sigmoid'),
             Reshape(output_shape)]
            ]
        self.build((None, )+input_shape)

    def call(self, inputs):
        x = self.flatten(inputs)
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))


class Discriminator(Model):
    '''判别器，判别生成样本和真实样本'''
    def __init__(self, 
                 input_shape, 
                 **kwargs):
        super().__init__(**kwargs)
        self.flatten = Flatten()
        self.blocks = [
            [Dense(512),
             LeakyReLU(alpha=0.2)],
            [Dense(256),
             LeakyReLU(alpha=0.2)],
             [Dense(1, activation='sigmoid')]
            ]
        self.build((None, )+input_shape)

    def call(self, inputs):
        x = self.flatten(inputs)
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))
   
        
class SimpleGAN():
    def __init__(
        self,
        latent_shape,
        output_shape,
        ):
        self.latent_shape = latent_shape
        self.output_shape = output_shape
        self.generator = Generator(
            input_shape=latent_shape,
            output_shape=output_shape
            )
        self.discriminator = Discriminator(
            input_shape=output_shape
            )
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics=['acc'])
        self.combined = self.combined_model()
        self.combined.compile(
            loss='binary_crossentropy', 
            optimizer=Adam(0.0002, 0.5),
            metrics=['acc'])

    def combined_model(self):
        _in = Input(self.latent_shape)
        x = self.generator(_in)
        self.discriminator.trainable = False
        x = self.discriminator(x)
        return Model(inputs=_in, outputs=x)

    def fit(
        self, 
        x, 
        epochs, 
        batch_size=128, 
        ):
        # 获取标签
        valid = K.ones((batch_size, 1))
        fake = K.zeros((batch_size, 1))
        
        for epoch in range(epochs):

            #--------------------------------训练判别器------------------------------------
            # 真实样本
            imgs = K.np.array(K.random.choices(x, k=batch_size), dtype='float32')
            # 采样随机噪声
            noise = K.random_uniform(((batch_size,)+self.latent_shape), 0, 1)
            # 生成样本
            gen_imgs = self.generator.predict(noise)
            # 更新参数
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_gen, d_acc_gen = self.discriminator.train_on_batch(gen_imgs, fake)
            #-----------------------------------------------------------------------------

            #--------------------------------训练生成器------------------------------------
            noise = K.random_uniform(((batch_size,)+self.latent_shape), 0, 1)
            g_loss, g_acc = self.combined.train_on_batch(noise, valid)
            #-----------------------------------------------------------------------------

            print (f" - epoch：{epoch} [D loss: {0.5*(d_loss_real+d_loss_gen)}, acc: {50*(d_acc_real+d_acc_gen)}%]")
            print (f" - epoch：{epoch} [G loss: {g_loss}, acc: {100*g_acc}%]")


    def save_weight(self, path):
        self.generator.save_weights(path+'generator.h5')
        self.discriminator.save_weights(path+'discriminator.h5')

    def load_weight(self, path):
        self.generator.load_weights(path+'generator.h5')
        self.discriminator.load_weights(path+'discriminator.h5')

    def sample_imgs(
        self, 
        batch_size, 
        save_path=None, 
        **kwargs):
        noise = K.random_uniform(((batch_size,)+self.latent_shape), 0, 1)
        gen_imgs = self.generator.predict(noise)
        gen_imgs = gen_imgs*255
        if save_path:
            for i in range(batch_size):
                plt.imsave(f'{save_path}{i}.jpeg', gen_imgs[i], **kwargs)
        return gen_imgs