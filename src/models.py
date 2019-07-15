from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, LeakyReLU, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K

from IPython.display import clear_output
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from umap import UMAP

from scanpy import read_h5ad
import matplotlib.pyplot as plt

def sampling(args):
	"""Reparameterization trick by sampling from an isotropic unit Gaussian.

		  instead of sampling from Q(z|X), sample epsilon = N(0,I)
		  z = z_mean + sqrt(var) * epsilon

		:param  args:
			mean and log of variance of Q(z|X)
	    :return:
		    sampled latent vector
	"""

	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# by default, random_normal has mean = 0 and std = 1.0
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon

##########################################
############### BASE MODEL ###############
##########################################
class Base():
	"""Base Model Class

	Attributes
	----------
	original_dim: int
		number of cells in the dataset
	latent_dim: int
		dimension of the latent space Z
	layers_dim: list
		array containing the dimension of all networks' dense layers:
		the first (second) [third] element is the list of the number of nodes in each
		layer of the encoder (decoder) [discriminator] deep neural network
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
	encoder: keras.engine.training.Model
		encoder deep neural network
	decoder: keras.engine.training.Model
		decoder deep neural network
	generator: keras.engine.training.Model
		generator deep neural network
	discriminator: keras.engine.training.Model
		discriminator deep neural network
	rec_loss: float
		reconstruction loss
	gen_loss: float
		generator loss
	dis_loss: float
		discriminator loss
	data: numpy.ndarray
		matrix containing gene expression
	gene_list: list
		list of gene names
	labels: list
		list of integers labelling cell subgroups

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	"""

	def __init__(self, latent_dim, layers_dim,
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
		self.encoder = None
		self.decoder = None
		self.generator = None
		self.discriminator = None
		self.rec_loss = None
		self.gen_loss = None
		self.dis_loss = None
		self.data = None
		self.gene_list = None
		self.labels = None

	def get_parameters(self):
		"""Print the list of network parameter.

		Returns
		-------
		Pandas dataframe object

		"""

		# construct a dictionary with the list of network parameters

		self.dict = {"Parameter": [], "Value": [], "Description": []}

		self.dict["Parameter"] = np.hstack(['latent_dim',
		                                    ['layer_' + str(k + 1) + '_enc_dim' for k in range(len(self.layers_dim[0]))],
		                                    ['layer_' + str(k + 1) + '_dec_dim' for k in range(len(self.layers_dim[1]))],
		                                    ['layer_' + str(k + 1) + '_dis_dim' for k in range(len(self.layers_dim[2]))],
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

		self.dict["Value"] = np.hstack([self.latent_dim,
		                                self.layers_dim[0],
		                                self.layers_dim[1],
		                                self.layers_dim[2],
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

		self.dict["Description"] = np.hstack(["dimension of latent space Z",
		                                      ["dimension of encoder dense layer " + str(k+1) for k in
		                                       range(len(self.layers_dim[0]))],
		                                      ["dimension of decoder dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_dim[1]))],
		                                      ["dimension of discriminator dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_dim[2]))],
		                                      "alpha coeff. in activation function",
		                                      "dropout rate",
		                                      "kernel initializer of all dense layers",
		                                      "bias initializer of all dense layers",
		                                      "weight of l2 kernel regularization",
		                                      "weight of l1 activity regularization",
		                                      "batch size",
		                                      "number of epochs",
		                                      "learning rate discriminator",
		                                      "learning rate generator",
		                                      "learning rate autoencoder",
		                                      "decay rate discriminator",
		                                      "decay rate generator",
		                                      "decay rate autoencoder"
		                                      ])

		return pd.DataFrame(self.dict, index=self.dict["Parameter"]).drop(columns=['Parameter'])

	def load_data(self, datapath):

		"""Read data file and initialize cell gene counts, gene name list and cell subgroups.

		Data file is a Scanpy AnnData object saved in h5ad format.
		This object contains cell subgoups obtained using Louvain algorithm.

		:param str datapath:
			path to data file (h5ad format)
		:return:
		:raises NameError: if Louvain clustering is not defined in the dataset
		"""

		adata = read_h5ad(datapath)

		self.data = adata.X

		self.original_dim = self.data.shape[1]

		self.gene_list = adata.var_names.values

		try:
			self.labels = adata.obs['louvain'].values.astype(int).tolist()
		except:
			print("Louvain clustering not defined in this dataset.")
			self.labels = None

		print("Dataset imported.")

	def rescale_data(self):

		"""Standardize gene expression counts by removing the mean and scaling to unit variance.

		:return:
		"""

		scaler = StandardScaler(with_mean=True, with_std=True)
		scaler.fit(self.data)
		self.data = scaler.transform(self.data)

		print("Gene expression data rescaled.")

	def get_summary(self):

		"""Print Adversarial Autoencoder model summary.

		:return:
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

	def export_graph(self, filepath):

		"""Save model graphs to PNG images.

		:param filepath:
			 path of output image files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		plot_model(self.encoder, to_file=filepath + "encoder.png", show_shapes=True)
		plot_model(self.decoder, to_file=filepath + "decoder.png", show_shapes=True)
		plot_model(self.autoencoder, to_file=filepath + "autoencoder.png", show_shapes=True)
		plot_model(self.generator, to_file=filepath + "generator.png", show_shapes=True)
		plot_model(self.discriminator, to_file=filepath + "discriminator.png", show_shapes=True)

		print("Model graphs saved.\n")

	def plot_umap(self, gene_selected, louvain=False):

		"""Plot the gene expression of selected genes in a 2-D latent space using the UMAP clustering algorithm.

		:param gene_selected:
			list of genes
		:type gene_selected: list
		:param louvain:
			if true, show label the 2-D latent space according Louvain clustering of cell data
		:type louvain: bool
		:return:
		"""

		z_mean, _, _ = self.encoder.predict(self.data, batch_size=self.batch_size)

		reducer = UMAP()
		z_mean = reducer.fit_transform(z_mean)

		for name in gene_selected:

			idx_name = np.where(self.gene_list == name)[0].tolist()[0]
			subset = self.data[:, idx_name]

			if louvain:

				plt.figure(figsize=(14, 5))

				plt.subplot(1, 2, 1)
				cmap = plt.get_cmap('viridis')  # RdBu
				plt.scatter(z_mean[:, 0], z_mean[:, 1],
				            c=subset,
				            cmap=cmap,
				            vmin=np.min(subset),
				            vmax=np.max(subset),
				            s=5)

				plt.colorbar()
				plt.title(name)
				plt.xlabel("z[0]")
				plt.ylabel("z[1]")

				plt.subplot(1, 2, 2)

				cmap2 = plt.get_cmap('tab20', np.max(self.labels) - np.min(self.labels) + 1)
				plt.scatter(z_mean[:, 0], z_mean[:, 1],
				            c=self.labels,
				            cmap=cmap2,
				            vmin=np.min(self.labels) - .5,
				            vmax=np.max(self.labels) + .5,
				            s=5)

				plt.colorbar()
				plt.title('Louvain Clustering')
				plt.xlabel("z[0]")
				plt.ylabel("z[1]")

				plt.tight_layout()
				plt.show()

			else:

				cmap = plt.get_cmap('viridis')  # RdBu
				plt.figure(figsize=(7, 5))
				plt.scatter(z_mean[:, 0], z_mean[:, 1],
				            c=subset,
				            cmap=cmap,
				            vmin=np.min(subset),
				            vmax=np.max(subset),
				            s=5)

				plt.colorbar()
				plt.title(name)
				plt.xlabel("z[0]")
				plt.ylabel("z[1]")

				plt.show()

	def export_model(self, filepath):

		"""Export the network models in h5 format.

		:param filepath:
			path of model files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		self.autoencoder.save(filepath + 'autoencoder.h5')
		self.generator.save(filepath + 'generator.h5')
		self.discriminator.save(filepath + 'discriminator.h5')
		self.encoder.save(filepath + 'encoder.h5')
		self.decoder.save(filepath + 'decoder.h5')

		print("All networks exported in h5 format.")


##########################################
############### MODEL n.1 ################
##########################################
class AAE1(Base):
	""" Unsupervised adversarial autoencoder model:
		reconstruction phase and regularization phase executed at each mini-batch.

	Methods
	-------
	build_model()
		build encoder, decoder, discriminator, generator and autoencoder architectures
	train(graph=False, gene=None)
		train the Adversarial Autoencoder
	"""

	def __init__(self, *args, **kwargs):
		super(AAE1, self).__init__(*args, **kwargs)

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer
		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim[0]):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          #kernel_regularizer=regularizers.l2(self.l2_weight),
			          #activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

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

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		decoder_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim[1]):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model(decoder_input, x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		discr_input = latent_input

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim[2]):
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

		"""Build generator neural network.

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

		# keep discriminator weights frozen
		discriminator.trainable = False

		generation = discriminator(compression)

		# instantiate and compile generator model
		generator = Model(input_encoder, generation)
		generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])

		return generator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		:return:
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

	def train(self, graph=False, gene=None):

		"""Training of the Adversarial Autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:return:
			lists containing reconstruction loss, generator loss, and discriminator loss at each epoch
		"""

		rec_loss = []
		gen_loss = []
		dis_loss = []

		val_split = 0.0

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(self.data)

			for i in range(int(len(self.data) / self.batch_size)):
				batch = self.data[i * self.batch_size:i * self.batch_size + self.batch_size]

				# Regularization phase
				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = np.concatenate([fake_pred, real_pred])
				discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                        np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=val_split,
				                                               verbose=0)

				# Reconstruction phase
				autoencoder_history = self.autoencoder.fit(x=batch,
				                                           y=batch,
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				generator_history = self.generator.fit(x=batch,
				                                       y=np.zeros(self.batch_size),
				                                       epochs=1,
				                                       batch_size=self.batch_size,
				                                       validation_split=val_split,
				                                       verbose=0)

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = generator_history.history["loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]

			if ((epoch + 1) % 10 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
						*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss])
				)

				if graph and (gene is not None):

					self.plot_umap(gene_selected=[gene])

			rec_loss.append(self.rec_loss)
			gen_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)

		print("Training completed.")

		return rec_loss, gen_loss, dis_loss


##########################################
############### MODEL n.2 ################
##########################################
class AAE2(Base):
	""" Model incorporating label information in the adversarial regularization.

	Methods
	-------
	build_model()
		build encoder, decoder, discriminator, generator and autoencoder architectures
	train(graph=False, gene=None)
		train the Adversarial Autoencoder

	"""

	def __init__(self, *args, **kwargs):
		super(AAE2, self).__init__(*args, **kwargs)

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer
		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim[0]):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          #kernel_regularizer=regularizers.l2(self.l2_weight),
			          #activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

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

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		decoder_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim[1]):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model(decoder_input, x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		labels_input = Input(shape=(labels_dim + 1,), name='Categories') # add one category for non-labeld data
		discr_input = concatenate([latent_input, labels_input])

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dim[2]):
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
		discriminator = Model([latent_input, labels_input], x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, input_encoder, compression, discriminator):

		"""Build generator neural network.

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

		# keep discriminator weights frozen
		discriminator.trainable = False

		generation = discriminator(compression)

		# instantiate and compile generator model
		generator = Model(input_encoder, generation)
		generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])

		return generator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		:return:
		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		encoder_input = Input(shape=(self.original_dim,), name='X')
		labels_input = Input(shape=(labels_dim + 1,), name='Categories')

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
		self.generator = self._build_generator([real_input, labels_input],
		                                       [self.encoder(real_input)[2], labels_input],
		                                       self.discriminator)

	def train(self, graph=False, gene=None):

		"""Training of the Adversarial Autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:return:
			lists containing reconstruction loss, generator loss, and discriminator loss at each epoch
		"""

		rec_loss = []
		gen_loss = []
		dis_loss = []

		val_split = 0.0

		labels_code = to_categorical(self.labels).astype(int)
		labels_zeros = np.zeros((len(self.labels), 1))  # extra label for training points with unknown classes
		labels_code = np.append(labels_code, labels_zeros, axis=1)

		data_ = np.concatenate([self.data, labels_code], axis=1)

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(data_)

			for i in range(int(len(self.data) / self.batch_size)):

				batch = data_[i * self.batch_size:i * self.batch_size + self.batch_size, :self.data.shape[1]]
				labels_ = data_[i * self.batch_size:i * self.batch_size + self.batch_size, self.data.shape[1]:]

				# Regularization phase
				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = [np.concatenate([fake_pred, real_pred]),
				                         np.concatenate([labels_, labels_])]
				discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                        np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=val_split,
				                                               verbose=0)

				# Reconstruction phase
				autoencoder_history = self.autoencoder.fit(x=batch,
				                                           y=batch,
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				generator_history = self.generator.fit(x=[batch, labels_],
				                                       y=np.zeros(self.batch_size),
				                                       epochs=1,
				                                       batch_size=self.batch_size,
				                                       validation_split=val_split,
				                                       verbose=0)

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = generator_history.history["loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]

			if ((epoch + 1) % 10 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
						*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss])
				)

				if graph and (gene is not None):

					self.plot_umap(gene_selected=[gene])

			rec_loss.append(self.rec_loss)
			gen_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)

		print("Training completed.")

		return rec_loss, gen_loss, dis_loss

