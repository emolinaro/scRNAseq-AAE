from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras import regularizers

import numpy as np
import os

from utils import sampling


#kernel_initializer =  RandomNormal(mean=0.0, stddev=0.01, seed=None)
#bias_initializer= RandomNormal(mean=0.0, stddev=0.01, seed=None)

kernel_initializer='glorot_uniform'
bias_initializer='zeros'


def build_encoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim):
    
    encoder_input = Input(shape=(original_dim, ), name="X")
    
    x = Dropout(rate=0.1, name='DO')(encoder_input)
    
    x = Dense(layer_1_dim, name="H1", use_bias = False,
              kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name='BN_1')(x)
    x = LeakyReLU(alpha=0.1, name='LR_1')(x)
    x = Dropout(rate=0.1, name='D1')(x)
    
    x = Dense(layer_2_dim, name="H2", use_bias = False,
              kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name='BN_2')(x)
    x = LeakyReLU(alpha=0.1, name='LR_2')(x)
    x = Dropout(rate=0.1, name='D2')(x)
    
    x = Dense(layer_3_dim, name="H3", use_bias = False,
              kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name='BN_3')(x)
    x = LeakyReLU(alpha=0.1, name='LR_3')(x)
    x = Dropout(rate=0.1, name='D3')(x)
    
    z_mean = Dense(latent_dim, name='z_mean', 
                   kernel_initializer=kernel_initializer, 
                   bias_initializer=bias_initializer)(x)
    z_log_var = Dense(latent_dim, name='z_log_var', 
                      kernel_initializer=kernel_initializer, 
                      bias_initializer=bias_initializer)(x)
    
    z = Lambda(sampling, output_shape=(latent_dim,), name='Z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    
    return encoder


def build_decoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim):
    
    decoder_input = Input(shape=(latent_dim,), name='Z')
    
    x = Dropout(rate=0.1, name='DO')(decoder_input)
    
    x = Dense(layer_3_dim, name="H1", use_bias = False,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    
    x = BatchNormalization(name='BN_1')(x)
    
    x = LeakyReLU(alpha=0.1, name='LR_1')(x)
    
    x = Dropout(rate=0.1, name='D1')(x)
    
    x = Dense(layer_2_dim, name="H2", use_bias = False,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    
    x = BatchNormalization(name='BN_2')(x)
    
    x = LeakyReLU(alpha=0.1, name='LR_2')(x)
    
    x = Dropout(rate=0.1, name='D2')(x)
    
    x = Dense(layer_1_dim, name="H3", use_bias = False,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01))(x)
    
    x = BatchNormalization(name='BN_3')(x)
    
    x = LeakyReLU(alpha=0.1, name='LR_3')(x)
    
    x = Dropout(rate=0.1, name='D3')(x)
    
    x = Dense(original_dim, activation='sigmoid', name="Xp")(x)

    # instantiate decoder model
    decoder = Model(decoder_input, x, name='decoder')
    
    return decoder

def build_VAE(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim):
    
    input_encoder = Input(shape=(original_dim, ), name='X')
    
    # build encoder
    encoder = build_encoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim)
    
    # build decoder
    decoder = build_decoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim)
    
    # instantiate VAE model
    outputs = decoder(encoder(input_encoder)[2])
    vae = Model(input_encoder, outputs, name='vae')
    
    # expected negative log-likelihood of the ii-th datapoint (reconstruction loss)
    reconstruction_loss = mse(input_encoder, outputs)
    #reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim

    # add regularizer: Kullback-Leibler divergence between the encoder’s distribution Q(z|x) and p(z)
    z_mean = encoder(input_encoder)[0]
    z_log_var = encoder(input_encoder)[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(lr=0.0002), metrics=['accuracy'])
    
    return encoder, decoder, vae

def train_VAE(vae, x_train, batch_size, epochs, val_split=0.2):
    
    vae_history = vae.fit(x_train, 
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_split=val_split,
                          verbose=1)
    
    return vae_history

    
    