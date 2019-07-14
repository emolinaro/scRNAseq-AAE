from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    
          instead of sampling from Q(z|X), sample epsilon = N(0,I)
          z = z_mean + sqrt(var) * epsilon
          
       Arguments
          args (tensor): mean and log of variance of Q(z|X)
       Returns
          z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 gene_list,
                 gene_names,
                 batch_size=None,
                 ):

    encoder, decoder = models
    
    x_test = data
    
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    
    for name in gene_list:
    
        idx_name = np.where(gene_names == name)[0].tolist()[0]
        y_test = x_test[:,idx_name]

        cmap = plt.get_cmap('RdBu')
        plt.figure(figsize=(8, 6))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap=cmap, vmin = np.min(y_test), vmax = np.max(y_test), s=10)
        plt.colorbar()
        plt.title(name)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()
        
        
def plot_results_louvain(models,
                         data,
                         labels,
                         batch_size=None,
                         reducer='UMAP'):

    encoder, decoder = models
    
    x_test = data
    y_test = labels
    
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    
    if reducer.lower() == 'umap':
        reducer = UMAP()
        z_mean = reducer.fit_transform(z_mean)
    
    else:
        n_pca = 10
        pca = PCA(n_components=n_pca)
        z_mean = pca.fit_transform(z_mean)
        
    
    cmap = plt.get_cmap('tab20', np.max(y_test)-np.min(y_test)+1)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap=cmap, vmin = np.min(y_test)-.5, vmax = np.max(y_test)+.5, s=10)
    plt.colorbar()
    plt.title('Louvain')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    
        
def plot_results_pca(models,
                     data,
                     gene_list,
                     gene_names,
                     latent_dim,
                     batch_size=None,
                     ):

    encoder, decoder = models
    
    x_test = data
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    
    n_pca = latent_dim
    pca = PCA(n_components=n_pca)
    z_mean = pca.fit_transform(z_mean)
    
    for name in gene_list:
    
        idx_name = np.where(gene_names == name)[0].tolist()[0]
        y_test = x_test[:,idx_name]
        
    
        cmap = plt.get_cmap('viridis')
        plt.figure(figsize=(8, 6))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap=cmap, vmin = np.min(y_test), vmax = np.max(y_test), s=10)
        plt.colorbar()
        plt.title(name)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()
        
        
def plot_results_umap(models,
                      data,
                      labels,
                      gene_list,
                      gene_names,
                      louvain=False,
                      batch_size=None):

    encoder, decoder = models
    
    x_test = data
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
        
    reducer = UMAP()
    z_mean = reducer.fit_transform(z_mean)
    
    for name in gene_list:
    
        idx_name = np.where(gene_names == name)[0].tolist()[0]
        y_test = x_test[:,idx_name]
        
        if louvain:
            
            plt.figure(figsize=(14, 5))
            
            plt.subplot(1,2,1)
            cmap = plt.get_cmap('viridis') # RdBu 
            plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                        c=y_test, 
                        cmap=cmap, 
                        vmin = np.min(y_test), 
                        vmax = np.max(y_test), 
                        s=10)
            
            plt.colorbar()
            plt.title(name)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            
            plt.subplot(1,2,2)
            
            cmap2 = plt.get_cmap('tab20', np.max(labels)-np.min(labels)+1)
            plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                        c=labels, 
                        cmap=cmap2, 
                        vmin = np.min(labels)-.5, 
                        vmax = np.max(labels)+.5, 
                        s=10)
            
            plt.colorbar()
            plt.title('Louvain')
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            
            plt.tight_layout()
            plt.show()
            
        else: 
        
            cmap = plt.get_cmap('viridis') # RdBu
            plt.figure(figsize=(7, 5))
            plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                        c=y_test, cmap=cmap, 
                        vmin = np.min(y_test), 
                        vmax = np.max(y_test), 
                        s=10)
            
            plt.colorbar()
            plt.title(name)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            
            plt.show()
            