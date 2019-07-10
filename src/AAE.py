from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, Activation, LeakyReLU
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import RandomNormal
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers

from IPython.display import clear_output

import numpy as np
import os, sys

from utils import sampling, plot_results_pca


#kernel_initializer =  RandomNormal(mean=0.0, stddev=0.01, seed=None)
#bias_initializer= RandomNormal(mean=0.0, stddev=0.01, seed=None)

kernel_initializer='glorot_uniform'
bias_initializer='zeros'


def build_encoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim):
    
    # TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

    # GAUSSIAN POSTERIOR
    
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

def build_discriminator(latent_dim, layer_1_dim, layer_2_dim, layer_3_dim):
    
    
    # build discriminator model
    
    optimizer_dis = Adam(lr=0.0001)

    discr_input = Input(shape=(latent_dim,), name='Z')
    
    x = Dropout(rate=0.1, name='DO')(discr_input)
    
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
    
    x = Dense(1, activation='sigmoid', name="Check")(x)

    # instantiate and compile decoder model
    discriminator = Model(discr_input, x, name='discriminator')
    discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])
    
    return discriminator


def build_generator(input_encoder, compression, discriminator):
    
    optimizer_gen = Adam(lr=0.0001)
    
    discriminator.trainable = False

    generation = discriminator(compression)
    
    # instantiate and compile generator model
    generator = Model(input_encoder, generation)
    generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])
    
    return generator

def build_AAE(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim):
    
    # compile the models
    optimizer_aae = Adam(lr=0.0002)
    
    #optimizer_aae = SGD(lr=0.001, decay=1e-6, momentum=0.9)
    
    input_encoder = Input(shape=(original_dim, ), name='X')
    
    # build encoder
    encoder = build_encoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim)
    
    # build decoder
    decoder = build_decoder(original_dim, latent_dim, layer_1_dim, layer_2_dim, layer_3_dim)
    
    # build and compile discriminator
    discriminator = build_discriminator(latent_dim, layer_1_dim, layer_2_dim, layer_3_dim)

    # ibuild and compile AAE model
    real_input = input_encoder
    compression = encoder(real_input)[2]
    reconstruction = decoder(compression)
 
    aae = Model(real_input, reconstruction, name='autoencoder')
    aae.compile(optimizer=optimizer_aae, loss='mse')
    
    # build and compile generator model
    generator = build_generator(real_input, compression, discriminator)
    
    return encoder, decoder, discriminator, generator, aae


def train_AAE(aae, generator, discriminator, encoder, decoder, x_train, batch_size, latent_dim, epochs, gene, gene_names, graph=False, val_split=0.0):
    
    rec_loss = []
    gen_loss = []
    disc_loss = []
    
    for epoch in range(epochs):
        np.random.shuffle(x_train)
    
        for i in range(int(len(x_train) / batch_size)):
        
            batch = x_train[i*batch_size:i*batch_size+batch_size]
            
            # Reconstruction phase
            #aae.train_on_batch(batch, [batch,np.ones(batch_size,1)])
            aae_history = aae.fit(x=batch, 
                                  y=batch, 
                                  epochs=1, 
                                  batch_size=batch_size, 
                                  validation_split=val_split,
                                  verbose=0)
    
            # Regularization phase
            fake_pred = encoder.predict(batch)[2]
            real_pred = np.random.normal(size=(batch_size,latent_dim)) # prior distribution
            discriminator_batch_x = np.concatenate([fake_pred, real_pred])
            discriminator_batch_y = np.concatenate([np.random.uniform(0.9,1.0,batch_size),
                                                    np.random.uniform(0.0,0.1,batch_size)])
        
            discriminator_history = discriminator.fit(x=discriminator_batch_x, 
                                                      y=discriminator_batch_y, 
                                                      epochs=1, 
                                                      batch_size=batch_size, 
                                                      validation_split=val_split,
                                                      verbose=0)
            
            # Freeze the discriminator weights during training of generator
            generator_history = generator.fit(x=batch, 
                                              y=np.zeros(batch_size), 
                                              epochs=1, 
                                              batch_size=batch_size, 
                                              validation_split=val_split,
                                              verbose=0)
        
        
            # check that the weights of disciminator and generator are the same and change at each batch
            #print(discriminator.get_weights()[0][0])
            #print("================================")
            #print(generator.get_layer('discriminator').get_weights()[0][0])
            #print("--------------------------------")
            
        
        if graph:
            if ((epoch+1)%1 == 0):

                clear_output()
                
                print("Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
                          *[epoch+1, epochs, aae_history.history["loss"][0], 
                            generator_history.history["loss"][0], 
                            discriminator_history.history["loss"][0]]))
                
                plot_results_pca((encoder, decoder), x_train, [gene], gene_names, latent_dim)
                
        else:
            if ((epoch+1)%10 == 0):
                
                #clear_output()
                
                print("Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
                          *[epoch+1, epochs, aae_history.history["loss"][0], 
                            generator_history.history["loss"][0], 
                            discriminator_history.history["loss"][0]]))
                                 
        rec_loss.append(aae_history.history["loss"][0])
        gen_loss.append(generator_history.history["loss"][0])
        disc_loss.append(discriminator_history.history["loss"][0])
        
    return rec_loss, gen_loss, disc_loss

    
    