from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, LeakyReLU, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import regularizers

from IPython.display import clear_output
from keras.utils import to_categorical

import numpy as np
import pandas as pd

from utils import sampling, plot_results_pca, plot_results_umap


class Base():
	"""Base Model Class

	Attributes
	----------
	original_dim: int
		number of cells in the dataset
	latent_dim: int
		dimension of the latent space Z
	layers_dim: list
		list containing the dimension of all networks' dense layers
	lr_alpha: float
		alpha parameter of LeackyReLU activation function
	do_rate: float
		dropout rate
	kernel_initializer: str
		kernel initializer of all dense layers
	bias_initializer: str
		bias initializer of all dense layers
	l2_weight: float
		weight parameter of l2 kernel regularization
	l1_weight: float
		weight parameter of l1 activity regularization
	batch_size: int
		batch size during training
	epochs: int
		number of epochs during training
	lr_dis: float
		learning rate discriminator optimizer
	lr_gen: float
		learning rate generator optimizer
	lr_ae: float
		learning rate autoencoder optimizer
	dr_dis: float
		decay rate discriminator optimizer
	dr_gen: float
		decay rate generator optimizer
	dr_ae: float
		decay rate autoencoder optimizer

	Methods
	-------
	get_parameters()
		Prints the animals name and what sound it makes

	"""

	def __init__(self, original_dim, latent_dim, layers_dim,
	             alpha=0.1, do_rate=0.1,
	             kernel_initializer='glorot_uniform',
	             bias_initializer='zeros',
	             l2_weight=0.01,
	             l1_weight=0.01,
	             batch_size=35,
	             epochs=50,
	             lr_dis=0.0001,
	             lr_gen=0.0001,
	             lr_ae=0.0002,
	             dr_dis=1e-6,
	             dr_gen=1e-6,
	             dr_ae=1e-6):
		self.original_dim = original_dim
		self.latent_dim = latent_dim
		self.layers_dim = layers_dim
		self.alpha = alpha
		self.do_rate = do_rate
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.l2_weight = l2_weight
		self.l1_weight = l1_weight
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr_dis = lr_dis
		self.lr_gen = lr_gen
		self.lr_ae = lr_ae
		self.dr_dis = dr_dis
		self.dr_gen = dr_gen
		self.dr_ae = dr_ae

	def get_parameters(self):
		"""Output the list of network parameter

		Returns
		-------
		Pandas dataframe object

		"""

		# construct a dictionary with the list of network parameters

		self.dict = {"Parameter": [], "Value": [], "Description": []}

		self.dict["Parameter"] = np.hstack(['original_dim',
		                                    'latent_dim',
		                                    ['layer_' + str(k + 1) + '_dim' for k in range(len(self.layers_dim))],
		                                    'alpha',
		                                    'do_rate',
		                                    'kernel_initializer',
		                                    'bias_initializer',
		                                    'l2_weight',
		                                    'l2_weight',
		                                    'batch_size',
		                                    'epochs',
		                                    'lr_dis',
		                                    'lr_gen',
		                                    'lr_ae',
		                                    'dr_dis',
		                                    'dr_gen',
		                                    'dr_ae'
		                                    ])

		self.dict["Value"] = np.hstack([self.original_dim,
		                                self.latent_dim,
		                                self.layers_dim,
		                                self.alpha,
		                                self.do_rate,
		                                self.kernel_initializer,
		                                self.bias_initializer,
		                                self.l2_weight,
		                                self.l1_weight,
		                                self.batch_size,
		                                self.epochs,
		                                self.lr_dis,
		                                self.lr_gen,
		                                self.lr_ae,
		                                self.dr_dis,
		                                self.dr_gen,
		                                self.dr_ae
		                                ])

		self.dict["Description"] = np.hstack(["number of cells",
		                                      "dimension of latent space Z",
		                                      ["dimension of dense layer " + str(k) for k in
		                                       range(len(self.layers_dim))],
		                                      "alpha coefficient in activation function",
		                                      "dropout rate",
		                                      "kernel initializer of all dense layers",
		                                      "bias initializer of all dense layers",
		                                      "weight of l2 kernel regularization",
		                                      "weight of l1 activity regularization",
		                                      "batch size",
		                                      "number of epochs",
		                                      "learning rate discriminator optimizer",
		                                      "learning rate generator optimizer",
		                                      "learning rate autoencoder optimizer",
		                                      "decay rate discriminator optimizer",
		                                      "decay rate generator optimizer",
		                                      "decay rate autoencoder optimizer"
		                                      ])

		return pd.DataFrame(self.dict, index=self.dict["Parameter"]).drop(columns=['Parameter'])


class AAE1(Base):
	""" Unsupervised Adversarial Autoencoder:

		reconstruction phase and regularization phase executed at each mini-batch

	Methods
	-------
	build_model()
		build encoder, decoder, discriminator, generator and autoencoder architectures
	get_summary(model)
		print model summary
	get_graph(model)
		print model graph
	save_graph(model, filename)
		save model graph to file
	"""

	def __init__(self, *args, **kwargs):
		super(AAE1, self).__init__(*args, **kwargs)

	def _build_encoder(self):

		"""Build encoder neural network

		:return:
			encoder
		"""

		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               kernel_initializer=self.kernel_initializer,
		               bias_initializer=self.bias_initializer)(x)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  kernel_initializer=self.kernel_initializer,
		                  bias_initializer=self.bias_initializer)(x)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network

		:return:
			decoder
		"""

		decoder_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		n_dense = len(self.layers_dim)

		# add dense layers
		for i, nodes in reversed(list(enumerate(self.layers_dim))):
			x = Dense(nodes,
			          name="H_" + str(n_dense - i),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(n_dense - i))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(n_dense - i))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(n_dense - i))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model(decoder_input, x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network

		:return:
			discriminator
		"""

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		discr_input = latent_input

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		# TODO: check performance wit inverted nodes
		for i, nodes in enumerate(self.layers_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(latent_input, x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, input_encoder, compression, discriminator):

		"""Build generator neural network

		:param input_encoder:
			encoder input layer
		:param compression:
			encoder transformation
		:param discriminator:
			initialized discriminator model
		:return:
			generator
		"""

		optimizer_gen = Adam(lr=self.lr_gen, decay=self.dr_gen)

		discriminator.trainable = False

		generation = discriminator(compression)

		# instantiate and compile generator model
		generator = Model(input_encoder, generation)
		generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])

		return generator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		# optimizer_aae = SGD(lr=0.001, decay=1e-6, momentum=0.9)

		encoder_input = Input(shape=(self.original_dim,), name='X')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile discriminator
		self.discriminator = self._build_discriminator()

		# build and compile autoencoder
		real_input = encoder_input
		compression = self.encoder(real_input)[2]
		reconstruction = self.decoder(compression)
		self.autoencoder = Model(real_input, reconstruction, name='autoencoder')
		self.autoencoder.compile(optimizer=optimizer_ae, loss='mse')

		# build and compile generator model
		self.generator = self._build_generator(real_input,
		                                       self.encoder(real_input)[2],
		                                       self.discriminator)

	def get_summary(self):

		"""Print Adversarial Autoencoder model summary

		"""

		print("\nEncoder Network")
		print("===============")
		self.encoder.summary()

		print("\nDecoder Network")
		print("===============")
		self.decoder.summary()

		print("\nAutoencoder Network")
		print("===================")
		self.autoencoder.summary()

		print("\nGenerator Network")
		print("=================")
		# Freeze the discriminator weights during training of generator
		self.generator.summary()

		print("\nDiscriminator Network")
		print("=====================")
		self.discriminator.trainable = True  # note: model already compiled with trainable weights
		self.discriminator.summary()

	def save_graph(self, filepath):

		"""Save model graphs to PNG images

		:param filepath:
			 path of output image files
		:type filepath: str
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		plot_model(self.encoder, to_file=filepath + "encoder_AAE1.png", show_shapes=True)
		plot_model(self.decoder, to_file=filepath + "decoder_AAE1.png", show_shapes=True)
		plot_model(self.autoencoder, to_file=filepath + "autoencoder_AAE1.png", show_shapes=True)
		plot_model(self.generator, to_file=filepath + "generator_AAE1.png", show_shapes=True)
		plot_model(self.discriminator, to_file=filepath + "discriminator_AAE1.png", show_shapes=True)

		print("Model graphs saved.\n")
