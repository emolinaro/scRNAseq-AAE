from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, LeakyReLU, Softmax, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras import regularizers
from keras import backend as K
from keras.activations import softmax
from keras.losses import mse
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers as L

from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from umap import UMAP

from scanpy import read_h5ad
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from os import makedirs
from keras.models import load_model


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

          Instead of sampling from Q(z|X), sample epsilon = N(0,I)
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


def sampling_gumbel(logits_y, tau):
    """Gumbel trick to sample from a discrete distribution.

    The original discrete distribution is replaced with a deterministic transformation of
    the stardard (parameterless) Gumbel distribution.

    :param logits_y:
        logits of the discrete distribution
    :type logit_y: tensorflow.python.framework.ops.Tensor

    :param tau:
        temperature parameter
    :type tau: float

    :return:
        tensor object with one-hot encoding of the discrete variable
    """

    G = K.random_uniform(K.shape(logits_y), 0, 1)

    # logits of y + Gumbel noise
    y = logits_y - K.log(-K.log(G + 1e-20) + 1e-20)

    # apply softmax to approximate the argmax function
    tau_var = K.variable(tau, name="temperature")

    # Gumbel-Softmax estimator
    y = K.softmax(y / tau_var)

    return y


##########################################
############### BASE MODEL ###############
##########################################
class Base():
    """Base Model Class.

        The class initialize parameters for the Adversarial and Variational Autoencoder Models.
        Overriding implemented for some of the methods in the child classes.

    Attributes
    ----------
    original_dim: int
        number of cells in the dataset
    latent_dim: int
        dimension of the latent space Z
    layers_enc_dim: list
        array containing the dimension of encoder network dense layers
    layers_dec_dim: list
        array containing the dimension of decoder network dense layers
    layers_dis_dim: list
        array containing the dimension of discriminator network dense layers
    layers_dis_cat_dim: list
        array containing the dimension of categorical discriminator network dense layers
    alpha: float
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
    lr_ae: float
        learning rate autoencoder optimizer
    dr_dis: float
        decay rate discriminator optimizer
    dr_ae: float
        decay rate autoencoder optimizer
    encoder: keras.engine.training.Model
        encoder deep neural network
    decoder: keras.engine.training.Model
        decoder deep neural network
    discriminator: keras.engine.training.Model
        discriminator deep neural network
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

    def __init__(self,
                 latent_dim=None,
                 layers_enc_dim=None,
                 layers_dec_dim=None,
                 layers_dis_dim=None,
                 alpha=0.1,
                 do_rate=0.1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 l2_weight=None,
                 l1_weight=None,
                 batch_size=35,
                 epochs=50,
                 lr_dis=0.0001,
                 lr_ae=0.0002,
                 dr_dis=1e-6,
                 dr_ae=1e-6):

        self.latent_dim = latent_dim
        self.layers_enc_dim = layers_enc_dim
        self.layers_dec_dim = layers_dec_dim
        self.layers_dis_dim = layers_dis_dim
        self.alpha = alpha
        self.do_rate = do_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        if self.l2_weight is None:
            self.kernel_regularizer = None
        else:
            self.kernel_regularizer = regularizers.l2(self.l2_weight)
        if self.l1_weight is None:
            self.activity_regularizer = None
        else:
            self.activity_regularizer = regularizers.l1(self.l1_weight)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_dis = lr_dis
        self.lr_ae = lr_ae
        self.dr_dis = dr_dis
        self.dr_ae = dr_ae
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.autoencoder = None
        self.data = None
        self.gene_list = None
        self.labels = None

        # create a dictionary with all networks' parameters
        self._get_parameters()

    def _get_parameters(self):
        """Create a dictionary with network parameter.

        """

        # construct a dictionary with the list of network parameters

        self.dict = {"Parameter": [], "Value": [], "Description": []}

        self.dict["Parameter"] = np.hstack(['batch_size',
                                            'epochs',
                                            'alpha',
                                            'do_rate',
                                            'kernel_initializer',
                                            'bias_initializer',
                                            'l2_weight',
                                            'l1_weight',
                                            'latent_dim',
                                            ['layer_' + str(k + 1) + '_enc_dim' for k in
                                             range(len(self.layers_enc_dim))],
                                            ['layer_' + str(k + 1) + '_dec_dim' for k in
                                             range(len(self.layers_dec_dim))],
                                            'lr_ae',
                                            'dr_ae',
                                            ['layer_' + str(k + 1) + '_dis_dim' for k in
                                             range(len(self.layers_dis_dim))],
                                            'lr_dis',
                                            'dr_dis',
                                            ])

        self.dict["Value"] = np.hstack([self.batch_size,
                                        self.epochs,
                                        self.alpha,
                                        self.do_rate,
                                        self.kernel_initializer,
                                        self.bias_initializer,
                                        self.l2_weight,
                                        self.l1_weight,
                                        self.latent_dim,
                                        self.layers_enc_dim,
                                        self.layers_dec_dim,
                                        self.lr_ae,
                                        self.dr_ae,
                                        self.layers_dis_dim,
                                        self.lr_dis,
                                        self.dr_dis,
                                        ])

        self.dict["Description"] = np.hstack(["batch size",
                                              "number of epochs",
                                              "alpha coeff. in activation function",
                                              "dropout rate",
                                              "kernel initializer of all dense layers",
                                              "bias initializer of all dense layers",
                                              "weight of l2 kernel regularization",
                                              "weight of l1 activity regularization",
                                              "dimension of latent space Z",
                                              ["dimension of encoder dense layer " + str(k + 1) for k in
                                               range(len(self.layers_enc_dim))],
                                              ["dimension of decoder dense layer " + str(k + 1) for k in
                                               range(len(self.layers_dec_dim))],
                                              "learning rate of autoencoder",
                                              "decay rate of autoencoder",
                                              ["dimension of discriminator dense layer " + str(k + 1) for k in
                                               range(len(self.layers_dis_dim))],
                                              "learning rate of discriminator",
                                              "decay rate of discriminator",
                                              ])

    def get_parameters(self):
        """Print the list of network parameter.

        Returns
        -------
        Pandas dataframe object
        """

        dataframe = pd.DataFrame(self.dict, index=self.dict["Parameter"]).drop(columns=['Parameter'])

        return dataframe

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
            # self.labels = None
            self.labels = np.zeros((self.original_dim,), dtype=int)

        print("Dataset imported.")

    def rescale_data(self):

        """Standardize gene expression counts by removing the mean and scaling to unit variance.

        :return:
        """

        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(self.data)
        self.data = scaler.transform(self.data)

        print("Dataset rescaled.")

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
            figure
        """

        z_mean = self.encoder.predict(self.data, batch_size=self.batch_size)[0]

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
        self.discriminator.save(filepath + 'discriminator.h5')
        self.encoder.save(filepath + 'encoder.h5')
        self.decoder.save(filepath + 'decoder.h5')

        print("All networks exported in h5 format.")


##########################################
############### VAE MODEL ################
##########################################
class VAE(Base):
    """ Unsupervised clustering with variational autoencoder model.

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

    Raises
    ------
    TypeError
        If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim.
    """

    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)

        if self.latent_dim is None or \
                self.layers_enc_dim is None or \
                self.layers_dec_dim is None:
            raise TypeError(
                "List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

        # create a dictionary with all networks' parameters
        self._get_parameters()

    def _get_parameters(self):
        """Create a dictionary with network parameter.

        """

        # construct a dictionary with the list of network parameters

        self.dict = {"Parameter": [], "Value": [], "Description": []}

        self.dict["Parameter"] = np.hstack(['batch_size',
                                            'epochs',
                                            'alpha',
                                            'do_rate',
                                            'kernel_initializer',
                                            'bias_initializer',
                                            'l2_weight',
                                            'l1_weight',
                                            'latent_dim',
                                            ['layer_' + str(k + 1) + '_enc_dim' for k in
                                             range(len(self.layers_enc_dim))],
                                            ['layer_' + str(k + 1) + '_dec_dim' for k in
                                             range(len(self.layers_dec_dim))],
                                            'lr_ae',
                                            'dr_ae'
                                            ])

        self.dict["Value"] = np.hstack([self.batch_size,
                                        self.epochs,
                                        self.alpha,
                                        self.do_rate,
                                        self.kernel_initializer,
                                        self.bias_initializer,
                                        self.l2_weight,
                                        self.l1_weight,
                                        self.latent_dim,
                                        self.layers_enc_dim,
                                        self.layers_dec_dim,
                                        self.lr_ae,
                                        self.dr_ae
                                        ])

        self.dict["Description"] = np.hstack(["batch size",
                                              "number of epochs",
                                              "alpha coeff. in activation function",
                                              "dropout rate",
                                              "kernel initializer of all dense layers",
                                              "bias initializer of all dense layers",
                                              "weight of l2 kernel regularization",
                                              "weight of l1 activity regularization",
                                              "dimension of latent space Z",
                                              ["dimension of encoder dense layer " + str(k + 1) for k in
                                               range(len(self.layers_enc_dim))],
                                              ["dimension of decoder dense layer " + str(k + 1) for k in
                                               range(len(self.layers_dec_dim))],
                                              "learning rate of autoencoder",
                                              "decay rate of autoencoder"
                                              ])

    def get_summary(self):

        """Print Variational Autoencoder model summary.

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

        print("Model graphs saved.\n")

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
        self.encoder.save(filepath + 'encoder.h5')
        self.decoder.save(filepath + 'decoder.h5')

        print("All networks exported in h5 format.")

    def _build_encoder(self):

        """Build encoder neural network.

        :return:
            encoder
        """
        # NB: no kernel regularizer and activity regularizer
        # TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

        # GAUSSIAN POSTERIOR

        encoder_input = Input(shape=(self.original_dim,), name="X")

        x = encoder_input

        # add dense layers
        for i, nodes in enumerate(self.layers_enc_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer
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

        x = decoder_input

        # add dense layers
        for i, nodes in enumerate(self.layers_dec_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      activity_regularizer=self.activity_regularizer
                      )(x)

            x = BatchNormalization(name='BN_' + str(i + 1))(x)

            x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

            x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

        x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

        # instantiate decoder model
        decoder = Model(decoder_input, x, name='decoder')

        return decoder

    def build_model(self):

        """Build Variational Autoencoder model architecture.

        """

        optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

        # TODO: implement optimizer_ae = SGD(lr=0.001, decay=1e-6, momentum=0.9)

        encoder_input = Input(shape=(self.original_dim,), name='X')

        # build encoder
        self.encoder = self._build_encoder()

        # build decoder
        self.decoder = self._build_decoder()

        # build and compile variational autoencoder
        real_input = encoder_input
        compression = self.encoder(real_input)[2]
        reconstruction = self.decoder(compression)

        self.autoencoder = Model(real_input, reconstruction, name='autoencoder')

        # expected negative log-likelihood of the ii-th datapoint (reconstruction loss)
        reconstruction_loss = mse(real_input, reconstruction)
        reconstruction_loss *= self.original_dim

        # add regularizer: Kullback-Leibler divergence between the encoder’s distribution Q(z|x) and p(z)
        z_mean = self.encoder(real_input)[0]
        z_log_var = self.encoder(real_input)[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        ae_loss = K.mean(reconstruction_loss + kl_loss)

        self.autoencoder.add_loss(ae_loss)
        self.autoencoder.compile(optimizer=optimizer_ae, metrics=['accuracy'])

    def train(self, val_split=0.2, log_dir="./results/"):

        """Training of the Variational Autoencoder.
        The training will stop if there is no change in the validation loss after 30 epochs.

        :param val_split:
            fraction of data used for validation
        :type val_split: float
        :param log_dir:
            directory with exported model files and tensorboard checkpoints
        :type log_dir: str

        :return:
            lists containing training loss and validation loss
        """

        if log_dir[-1] != "/":
            log_dir = log_dir + "/"

        print("Start model training...")

        makedirs(log_dir + 'logs/summaries/', exist_ok=True)
        tensorboard = TensorBoard(log_dir=log_dir + 'logs/summaries/')

        vae_history = self.autoencoder.fit(self.data,
                                           epochs=self.epochs,
                                           batch_size=self.batch_size,
                                           validation_split=val_split,
                                           callbacks=[  # checkpoint('../models/vae_weights.hdf5'),
                                               EarlyStopping(monitor='val_loss',
                                                             patience=30,
                                                             verbose=0,
                                                             mode='min',
                                                             baseline=None,
                                                             restore_best_weights=False),
                                               tensorboard
                                           ],
                                           verbose=1)

        print("Training completed.")

        # save models in h5 format
        # this is a workaround to avoid AttributrError due to multiple putputs of encoder net
        # the same trick is applied in the other models

        makedirs(log_dir + 'models/', exist_ok=True)
        self.export_model(log_dir + 'models/')

        makedirs(log_dir + '/logs/projector/', exist_ok=True)
        with open(join(log_dir + 'logs/projector/', 'metadata.tsv'), 'w') as f:
            np.savetxt(f, self.labels, fmt='%i')

        self.encoder = load_model(log_dir + 'models/encoder.h5')

        tensorboard = TensorBoard(log_dir=log_dir + 'logs/projector/',
                                  batch_size=self.batch_size,
                                  embeddings_freq=1,
                                  write_graph=False,
                                  embeddings_layer_names=['z_mean', 'Z'],
                                  embeddings_metadata='metadata.tsv',
                                  embeddings_data=self.data
                                  )

        data_compression = self.encoder.predict(self.data, batch_size=self.batch_size)[2]
        self.encoder.compile(optimizer='adam', loss=[None, None, 'mse'])
        self.encoder.fit(self.data,
                         data_compression,
                         batch_size=self.batch_size,
                         callbacks=[tensorboard],
                         epochs=1,
                         verbose=0)
        print("Latent space embedding completed.")


        loss = vae_history.history["loss"]
        val_loss = vae_history.history["val_loss"]

        return loss, val_loss


##########################################
############### MODEL n.1 ################
##########################################
class AAE1(Base):
    """ Unsupervised adversarial autoencoder model.

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

    Raises
    ------
    TypeError
        If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim.
    """

    def __init__(self, **kwargs):
        super(AAE1, self).__init__(**kwargs)

        if self.latent_dim is None or \
                self.layers_enc_dim is None or \
                self.layers_dec_dim is None or \
                self.layers_dis_dim is None:
            raise TypeError(
                "List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

    def _build_encoder(self):

        """Build encoder neural network.

        :return:
            encoder
        """
        # NB: no kernel regularizer and activity regularizer
        # TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

        # GAUSSIAN POSTERIOR

        encoder_input = Input(shape=(self.original_dim,), name="X")

        x = encoder_input

        # add dense layers
        for i, nodes in enumerate(self.layers_enc_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer
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

        x = decoder_input

        # add dense layers
        for i, nodes in enumerate(self.layers_dec_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      activity_regularizer=self.activity_regularizer
                      )(x)

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
        optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

        latent_input = Input(shape=(self.latent_dim,), name='Z')
        discr_input = latent_input

        x = discr_input

        # add dense layers
        for i, nodes in enumerate(self.layers_dis_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      activity_regularizer=self.activity_regularizer
                      )(x)

            x = BatchNormalization(name='BN_' + str(i + 1))(x)

            x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

            x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

        x = Dense(1, activation='sigmoid', name="Check")(x)

        # instantiate and compile discriminator model
        discriminator = Model(latent_input, x, name='discriminator')
        discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

        return discriminator

    def _build_generator(self, compression, discriminator):

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

        # keep discriminator weights frozen
        discriminator.trainable = False

        generation = discriminator(compression)

        return generation

    def build_model(self):

        """Build Adversarial Autoencoder model architecture.

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

        # build generator
        compression = self.encoder(encoder_input)[2]
        generation = self._build_generator(compression, self.discriminator)

        # build and compile autoencoder
        reconstruction = self.decoder(compression)
        self.autoencoder = Model(encoder_input,
                                 [reconstruction, generation],
                                 name='autoencoder')
        self.autoencoder.compile(optimizer=optimizer_ae,
                                 loss=['mse', 'binary_crossentropy'],
                                 loss_weights=[0.99, 0.01]
                                 )

    def train(self, graph=False, gene=None, log_dir="./results"):

        """Training of the Adversarial Autoencoder.

        During the reconstruction phase the training of the generator proceeds with the
        discriminator weights frozen.

        :param graph:
            if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
        :type graph: bool
        :param gene:
            selected gene
        :type gene: str
        :param log_dir:
            directory with exported model files and tensorboard checkpoints
        :type log_dir: str

        :return:
            lists containing reconstruction loss, generator loss, and discriminator loss at each epoch
        """

        rec_loss = []
        dis_loss = []

        if log_dir[-1] != "/":
            log_dir = log_dir + "/"

        steps = int(len(self.data) / self.batch_size)
        batches = self.epochs * steps

        for step in range(batches):

            ids = np.random.randint(0, self.data.shape[0], self.batch_size)
            batch = self.data[ids]

            # Regularization phase
            fake_pred = self.encoder.predict(batch)[2]
            real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
            discriminator_batch_x = np.concatenate([fake_pred, real_pred])
            discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
                                                    np.random.uniform(0.0, 0.1, self.batch_size)])

            discriminator_history = self.discriminator.train_on_batch(x=discriminator_batch_x,
                                                                      y=discriminator_batch_y)
            dis_loss.append(discriminator_history[0])

            # Reconstruction phase
            real = np.zeros((self.batch_size,), dtype=int)
            autoencoder_train_history = self.autoencoder.train_on_batch(x=batch,
                                                                        y=[batch, real])

            rec_loss.append(autoencoder_train_history[0])

            if ((step+1) % steps == 0):

                clear_output(wait=True)

                print(
                    "Epoch {0:d}/{1:d}, rec. loss: {2:.6f}, dis. loss: {3:.6f}"
                        .format(
                        *[int((step + 1)/steps), self.epochs, rec_loss[0], dis_loss[0]])
                )

                if graph and (gene is not None):
                    self.plot_umap(gene_selected=[gene], louvain=True)

        print("Training completed.")

        # save models in h5 format
        makedirs(log_dir + 'models/', exist_ok=True)
        self.export_model(log_dir + 'models/')

        makedirs(log_dir + '/logs/projector/', exist_ok=True)
        with open(join(log_dir + 'logs/projector/', 'metadata.tsv'), 'w') as f:
            np.savetxt(f, self.labels, fmt='%i')

        self.encoder = load_model(log_dir + 'models/encoder.h5')

        tensorboard = TensorBoard(log_dir=log_dir + 'logs/projector/',
                                  batch_size=self.batch_size,
                                  embeddings_freq=1,
                                  write_graph=False,
                                  embeddings_layer_names=['z_mean', 'Z'],
                                  embeddings_metadata='metadata.tsv',
                                  embeddings_data=self.data
                                  )

        data_compression = self.encoder.predict(self.data, batch_size=self.batch_size)[2]
        self.encoder.compile(optimizer='adam', loss=[None, None, 'mse'])
        self.encoder.fit(self.data,
                         data_compression,
                         batch_size=self.batch_size,
                         callbacks=[tensorboard],
                         epochs=1,
                         verbose=0)
        print("Latent space embedding completed.")

        return rec_loss, dis_loss


##########################################
############### MODEL n.2 ################
##########################################
class AAE2(Base):
    """ Unsupervised adversarial autoencoder model with arebitrary number of clusters.

    Attributes
    ----------
    clusters: int
        number of clusters in the dataset
    tau: float
        temperature parameter used in the Gumbel-softmax trick
    layers_dis_cat_dim: list
        array containing the dimension of categorical discriminator network dense layers
    lr_dis_cat: float
        learning rate for categorical discriminator optimizer
    dr_dis_cat: float
        decay rate categorical discriminator optimizer
    discriminator_cat: keras.engine.training.Model
        categorical discriminator deep neural network

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

    Raises
    ------
    TypeError
        If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim, layers_dis_cat_dim.
    """

    def __init__(self,
                 num_clusters=None,
                 tau=0.5,
                 layers_dis_cat_dim=None,
                 lr_dis_cat=0.0001,
                 dr_dis_cat=1e-6,
                 **kwargs):

        self.num_clusters = num_clusters
        self.tau = tau
        self.layers_dis_cat_dim = layers_dis_cat_dim
        self.lr_dis_cat = lr_dis_cat
        self.dr_dis_cat = dr_dis_cat

        Base.__init__(self, **kwargs)

        if self.latent_dim is None or \
                self.layers_enc_dim is None or \
                self.layers_dec_dim is None or \
                self.layers_dis_dim is None or \
                self.layers_dis_cat_dim is None or \
                self.num_clusters is None:
            raise TypeError(
                "List of mandatory arguments: num_clusters, latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

        self.discriminator_cat = None
        self.dis_cat_loss = None

        # update dictionary of internal parameters
        self._update_dict()

    def _update_dict(self):
        """Update model dictionary of input parameters.

        """

        dict2 = {"Parameter": [], "Value": [], "Description": []}

        dict2["Parameter"] = np.hstack(
            [['layer_' + str(k + 1) + '_dis_cat_dim' for k in range(len(self.layers_dis_cat_dim))],
             'lr_dis_cat',
             'dr_dis_cat',
             'tau',
             'num_clusters'
             ])

        dict2["Value"] = np.hstack([self.layers_dis_cat_dim,
                                    self.lr_dis_cat,
                                    self.dr_dis_cat,
                                    self.tau,
                                    self.num_clusters
                                    ])

        dict2["Description"] = np.hstack([["dimension of cat. discriminator dense layer " + str(k + 1) for k in
                                           range(len(self.layers_dis_cat_dim))],
                                          "learning rate of cat. discriminator",
                                          "decay rate of cat. discriminator",
                                          "temperature parameter",
                                          "number of clusters in the dateset"
                                          ])

        for k in self.dict.keys():
            self.dict[k] = np.append(self.dict[k], dict2[k])

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

        print("\nDiscriminator Network")
        print("=====================")
        self.discriminator.trainable = True  # note: model already compiled with trainable weights
        self.discriminator.summary()

        print("\nCategorical Discriminator Network")
        print("=================================")
        self.discriminator_cat.trainable = True  # note: model already compiled with trainable weights
        self.discriminator_cat.summary()

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
        plot_model(self.discriminator, to_file=filepath + "discriminator.png", show_shapes=True)
        plot_model(self.discriminator_cat, to_file=filepath + "discriminator_cat.png", show_shapes=True)

        print("Model graphs saved.\n")

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
        self.discriminator.save(filepath + 'discriminator.h5')
        self.discriminator_cat.save(filepath + 'discriminator_cat.h5')
        self.encoder.save(filepath + 'encoder.h5')
        self.decoder.save(filepath + 'decoder.h5')

        print("All networks exported in h5 format.")

    def _build_encoder(self):

        """Build encoder neural network.

        :return:
            encoder
        """
        # NB: no kernel regularizer and activity regularizer

        # GAUSSIAN POSTERIOR

        encoder_input = Input(shape=(self.original_dim,), name="X")

        #x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)
        x = encoder_input

        # add dense layers
        for i, nodes in enumerate(self.layers_enc_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer
                      )(x)

            x = BatchNormalization(name='BN_' + str(i + 1))(x)

            x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

            x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

        z0 = Dense(self.layers_enc_dim[-1],
                   name="H_z",
                   kernel_initializer=self.kernel_initializer
                   )(x)

        z0 = BatchNormalization(name='BN_z')(z0)

        z0 = LeakyReLU(alpha=self.alpha, name='LR_z')(z0)

        z0 = Dropout(rate=self.do_rate, name='D_z')(z0)

        z_mean = Dense(self.latent_dim,
                       name='z_mean',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer
                       )(z0)

        z_log_var = Dense(self.latent_dim,
                          name='z_log_var',
                          kernel_initializer=self.kernel_initializer,
                          bias_initializer=self.bias_initializer
                          )(z0)

        z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

        y0 = Dense(self.layers_enc_dim[-1],
                   name="H_y",
                   kernel_initializer=self.kernel_initializer)(x)

        y0 = BatchNormalization(name='BN_y')(y0)

        y0 = LeakyReLU(alpha=self.alpha, name='LR_y')(y0)

        y0 = Dropout(rate=self.do_rate, name='D_y')(y0)

        y = Dense(self.num_clusters,
                  name='logits',
                  kernel_initializer=self.kernel_initializer,
                  bias_initializer=self.bias_initializer
                  )(y0)

        y = Lambda(sampling_gumbel, arguments={'tau': self.tau}, output_shape=(self.num_clusters,), name='y')(y)

        # instantiate encoder model
        encoder = Model(encoder_input, [z_mean, z_log_var, z, y], name='encoder')

        return encoder

    def _build_decoder(self):

        """Build decoder neural network.

        :return:
            decoder
        """

        latent_input = Input(shape=(self.latent_dim,), name='Z')
        classes = Input(shape=(self.num_clusters,), name='y')

        decoder_input = concatenate([latent_input, classes], name='Z_y')
        x = decoder_input
        # x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

        # add dense layers
        for i, nodes in enumerate(self.layers_dec_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      activity_regularizer=self.activity_regularizer
                      )(x)

            x = BatchNormalization(name='BN_' + str(i + 1))(x)

            x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

            x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

        x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

        # instantiate decoder model
        decoder = Model([latent_input, classes], x, name='decoder')

        return decoder

    def _build_discriminator(self):

        """Build discriminator neural network.

        :return:
            discriminator
        """

        optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

        discr_input = Input(shape=(self.latent_dim,), name='Z')

        # x = Dropout(rate=self.do_rate, name='D_O')(discr_input)
        x = discr_input

        # add dense layers
        for i, nodes in enumerate(self.layers_dis_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      activity_regularizer=self.activity_regularizer
                      )(x)

            x = BatchNormalization(name='BN_' + str(i + 1))(x)

            x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

            x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

        x = Dense(1, activation='sigmoid', name="Check")(x)

        # instantiate and compile discriminator model
        discriminator = Model(discr_input, x, name='discriminator')
        discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

        return discriminator

    def _build_generator(self, compression, compression_cat, discriminator, discriminator_cat):

        """Build generator neural network.

        :param compression:
            encoder transformation
        :param discriminator:
            initialized discriminator model
        :return:
            generator
        """

        # keep discriminator weights frozen
        discriminator.trainable = False
        discriminator_cat.trainable = False

        generation = discriminator(compression)
        generation_cat = discriminator_cat(compression_cat)

        return generation, generation_cat

    def _build_discriminator_cat(self):

        """Build categorical discriminator neural network.

        :return:
            discriminator_cat

        """
        optimizer_dis = Adam(lr=self.lr_dis_cat, decay=self.dr_dis_cat)

        discr_input = Input(shape=(self.num_clusters,), name='y')

        # x = Dropout(rate=self.do_rate, name='D_O')(discr_input)
        x = discr_input

        # add dense layers
        for i, nodes in enumerate(self.layers_dis_cat_dim):
            x = Dense(nodes,
                      name="H_" + str(i + 1),
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=self.kernel_regularizer,
                      activity_regularizer=self.activity_regularizer
                      )(x)

            x = BatchNormalization(name='BN_' + str(i + 1))(x)

            x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

            x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

        x = Dense(1, activation='sigmoid', name="Check")(x)

        # instantiate and compile discriminator model
        discriminator = Model(discr_input, x, name='discriminator_cat')
        discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

        return discriminator

    def build_model(self):

        """Build Adversarial Autoencoder model architecture.

        """

        optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

        encoder_input = Input(shape=(self.original_dim,), name='X')

        # build encoder
        self.encoder = self._build_encoder()

        # build decoder
        self.decoder = self._build_decoder()

        # build and compile discriminator
        self.discriminator = self._build_discriminator()

        # build and compile categorical discriminator
        self.discriminator_cat = self._build_discriminator_cat()

        # build generators
        compression = [self.encoder(encoder_input)[2], self.encoder(encoder_input)[3]]
        generation, generation_cat = self._build_generator(compression[0],
                                                           compression[1],
                                                           self.discriminator,
                                                           self.discriminator_cat)

        # build and compile autoencoder
        reconstruction = self.decoder(compression)
        self.autoencoder = Model(encoder_input,
                                 [reconstruction, generation, generation_cat],
                                 name='autoencoder')
        self.autoencoder.compile(optimizer=optimizer_ae,
                                 loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
                                 loss_weights=[0.99, 0.005, 0.005]
                                 )

    def train(self, graph=False, gene=None, log_dir="./results/"):

        """Training of the semisupervised adversarial autoencoder.

        During the reconstruction phase the training of the generator proceeds with the
        discriminator weights frozen.

        :param graph:
            if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
        :type graph: bool
        :param gene:
            selected gene
        :type gene: str
        :param val_split:
            fraction of the training data to be used as validation data
        :type val_split: float
        :param log_dir:
            directory with exported model files and tensorboard checkpoints
        :type log_dir: str

        :return:
            lists containing reconstruction training loss, reconstruction validation loss,
            discriminator loss, and categorical discriminator loss at each epoch
        """

        rec_loss = []
        val_loss = []
        dis_loss = []
        dis_cat_loss = []

        if log_dir[-1] != "/":
            log_dir = log_dir + "/"

        print("Start model training...")

        steps = int(len(self.data) / self.batch_size)
        batches = self.epochs * steps

        for step in range(batches):

            ids = np.random.randint(0, self.data.shape[0], self.batch_size)
            batch = self.data[ids]

            # Regularization phase
            real = np.random.uniform(0.0, 0.1, self.batch_size)
            fake = np.random.uniform(0.9, 1.0, self.batch_size)

            fake_pred = self.encoder.predict(batch)[2]
            real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
            discriminator_batch_x = np.concatenate([fake_pred, real_pred])
            discriminator_batch_y = np.concatenate([fake, real])


            discriminator_history = self.discriminator.train_on_batch(x=discriminator_batch_x,
                                                                      y=discriminator_batch_y)
            dis_loss.append(discriminator_history[0])

            fake_pred_cat = self.encoder.predict(batch)[3]
            class_sample = np.random.randint(low=0, high=self.num_clusters, size=self.batch_size)
            real_pred_cat = to_categorical(class_sample, num_classes=self.num_clusters).astype(int)

            discriminator_cat_batch_x = np.concatenate([fake_pred_cat, real_pred_cat])
            discriminator_cat_batch_y = np.concatenate([fake, real])

            discriminator_cat_history = self.discriminator_cat.train_on_batch(x=discriminator_cat_batch_x,
                                                                              y=discriminator_cat_batch_y)
            dis_cat_loss.append(discriminator_cat_history[0])

            # Reconstruction phase
            real = np.zeros((self.batch_size,), dtype=int)
            autoencoder_train_history = self.autoencoder.train_on_batch(x=batch,
                                                                        y=[batch, real, real])
            rec_loss.append(autoencoder_train_history[0])

            autoencoder_test_history = self.autoencoder.test_on_batch(x=batch,
                                                                      y=[batch, real, real])
            val_loss.append(autoencoder_test_history[0])

            if ((step+1) % steps == 0):

                clear_output(wait=True)

                print(
                    "Epoch {0:d}/{1:d}, rec. loss: {2:.6f}, val. loss: {3:.6f}, dis. loss: {4:.6f}, cat. dis. loss: {5:.6f}"
                        .format(
                        *[int((step + 1)/steps), self.epochs, rec_loss[0], val_loss[0], dis_loss[0], dis_cat_loss[0]])
                )

                if graph and (gene is not None):
                    self.plot_umap(gene_selected=[gene], louvain=True)

        print("Training completed.")

        # save models in h5 format
        makedirs(log_dir + 'models/', exist_ok=True)
        self.export_model(log_dir + 'models/')

        makedirs(log_dir + '/logs/projector/', exist_ok=True)
        with open(join(log_dir + 'logs/projector/', 'metadata.tsv'), 'w') as f:
            np.savetxt(f, self.labels, fmt='%i')

        self.encoder = load_model(log_dir + 'models/encoder.h5')

        tensorboard = TensorBoard(log_dir=log_dir + 'logs/projector/',
                                  batch_size=self.batch_size,
                                  embeddings_freq=1,
                                  write_graph=False,
                                  embeddings_layer_names=['z_mean', 'Z'],
                                  embeddings_metadata='metadata.tsv',
                                  embeddings_data=self.data
                                  )

        res = self.encoder.predict(self.data, batch_size=self.batch_size)
        data_compression = res[2]
        categories = res[3]
        self.encoder.compile(optimizer='adam',
                             loss=[None, None, 'mse', 'binary_crossentropy'])
        self.encoder.fit(self.data,
                         [data_compression, categories],
                         batch_size=self.batch_size,
                         callbacks=[tensorboard],
                         epochs=1,
                         verbose=0)
        print("Latent space embedding completed.")

        return rec_loss, val_loss, dis_loss, dis_cat_loss


