import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid


class PColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# The following class is a workaround to solve the following error when using Tensorboard:
# AttributeError: 'TensorBoard' object has no attribute 'sess'
# Issue reported here:
# https://github.com/tensorflow/tensorboard/issues/1666

class TensorBoardWithSession(tf.keras.callbacks.TensorBoard):

    def __init__(self, **kwargs):
        from tf.python.keras import backend as K
        self.sess = K.get_session()

        super().__init__(**kwargs)


def cluster_grid_scan(model, labels_true, res, n_nbs, n_pcs):
    """Perform a grid search of clusters similarity algorithms. Included measures are:
           1) adjusted random index (ARI)
           2) adjusted mutual information (AMI)
           3) cluster accuracy based on Hungarian maximum matching algorithm (CA)

    :param model:
        trained model (VAE, AAE1, AAE2)
    :param labels_true:
        ground truth labels
    :param res:
        resolution
    :param n_nbs:
        number of neighbors
    :param n_pcs:
        number of partial components
    :return:
        dictionary with grid parameters and cluster measures
    """

    param_grid = {'res': res, 'n_nbs': n_nbs, 'n_pcs': n_pcs}

    grid = ParameterGrid(param_grid)

    out = {'res': [], 'n_nbs': [], 'n_pcs': [], 'ARI': [], 'AMI': [], 'CA': []}

    for ps in tqdm(grid):

        model.update_labels(res=ps['res'], n_neighbors=ps['n_nbs'], n_pcs=ps['n_pcs'])
        labels_pred = model.labels
        _, ps['ARI'], ps['AMI'], ps['CA'] = model.eval_clustering(labels_true,
                                                                  labels_pred,
                                                                  graph=False,
                                                                  verbose=False)

        for item in ps.keys():
            out[item].append(ps[item])

    for item in out.keys():
        out[item] = np.array(out[item])

    return out


# Helperfunctions to make your feature definition more readable
def _int64_feature(value):
    """
        Returns an int64_list from a bool / enum / int / uint.

    """
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
    )


def _floats_feature(value):
    """
        Returns a float_list from a float / double.

    """
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
    )


def _bytes_feature(value):
    """
        Returns a bytes_list from a string / byte.

    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )


def export_to_tfrecord(filepath, adata, val_split=0.2):
    """Export dataset from h5ad format to TFRecord format.

       :param filepath:
              path of output file
       :type filepath: str

       :param adata:
              an annotated data matrix
       :type adata: AnnData object

       :param val_split:
              percentage of data assigned to validation dataset

       :return: TFRecord formatted files
    """

    data = adata.X

    dim = data.shape[0]

    train_size = int(dim * (1 - val_split))
    val_size = dim - train_size

    labels = adata.obs['louvain'].values.astype(int)

    # create train dataset
    writer = tf.python_io.TFRecordWriter(filepath + ".train")
    for i in range(train_size):
        feature = {'data': _bytes_feature(tf.compat.as_bytes(data[i].tostring())),
                   'label': _int64_feature(int(labels[i]))}

        # Serialize to string and write to file
        single = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(single.SerializeToString())

    # create train dataset
    writer = tf.python_io.TFRecordWriter(filepath + ".val")
    for i in range(val_size):
        feature = {'data': _bytes_feature(tf.compat.as_bytes(data[train_size + i].tostring())),
                   'label': _int64_feature(int(labels[train_size + i]))}

        # Serialize to string and write to file

        single = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(single.SerializeToString())


def data_generator(filepath, batch_size=35, epochs=200, num_cpus=24, is_training=True, auto_shard=False):
    """Build data pipeline.

    :param filepath:
        input file path in TFRecord format
    :param batch_size:
        batch size
    :param data_size:
        number of data entries
    :param epochs:
        number of epochs
    :param num_cpus:
        number of CPU cores to bbe used for pipeline parallelization
    :return:
        batch generator data pipeline
    """

    def _parse_function(proto):
        """Parse TFExample records.

        :param proto:
            single data entry
        :return:
            transformed data entry
        """
        # data fetures
        keys_to_features = {'data': tf.io.FixedLenFeature((), tf.string, ""),
                            'label': tf.io.FixedLenFeature((), tf.int64, -1)}

        # Load one example
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)

        # turn saved data string into an array
        parsed_features['data'] = tf.io.decode_raw(parsed_features['data'], tf.float32)

        return parsed_features['data']

    # load TFRecords and create a Dataset object
    files = tf.data.Dataset.list_files(filepath)
    # dataset = files.interleave(tf.data.TFRecordDataset)

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_cpus)

    # set the number of datapoints you want to load and shuffle
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    # repeat the dataset
    dataset = dataset.repeat(epochs)

    # parallelize data transformation
    dataset = dataset.map(_parse_function, num_parallel_calls=num_cpus)

    # set the batchsize
    dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)

    # prefetch elements from the input dataset ahead of the time they are requested
    dataset = dataset.prefetch(buffer_size=1)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard = auto_shard
    dataset = dataset.with_options(options)

    return dataset
