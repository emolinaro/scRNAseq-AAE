import tensorflow as tf
import os, sys
import argparse
import json
import textwrap

sys.path.append('../src')
from models import *
from utils import PColors as PC

import warnings

warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def header():
    version = '1.0.0'
    program = 'Single Cell RNA Sequencing Analysis'
    print(" ")
    header = PC.HEADER + """
    ============================================
    {} (v{}) 
    ============================================
    """.format(program, version) + PC.ENDC

    list = textwrap.wrap(header, width=50)
    for element in list:
        print(element)
    print("\n")


def init_model(param_file, model_type):
    # Model selection
    if model_type not in ['VAE', 'AAE1', 'AAE2']:
        print(PC.WARNING + "model type must be one of 'VAE', 'AAE1', and 'AAE2'." + PC.ENDC)
        sys.exit()

    # Initialize network parameters
    parameters = init_parameters(param_file, model_type)

    if model_type == 'VAE':
        model = VAE(**parameters)
        print(PC.RED + " =======================================  " + PC.ENDC)
        print(PC.RED + " Selected Model: Variational Autoencoder  " + PC.ENDC)
        print(PC.RED + " =======================================\n" + PC.ENDC)

    elif model_type == 'AAE1':
        model = AAE1(**parameters)
        print(PC.RED + " =========================================  " + PC.ENDC)
        print(PC.RED + " Selected Model: Adversarial Autoencoder 1  " + PC.ENDC)
        print(PC.RED + " =========================================\n" + PC.ENDC)

    elif model_type == 'AAE2':
        model = AAE2(**parameters)
        print(PC.RED + " =========================================  " + PC.ENDC)
        print(PC.RED + " Selected Model: Adversarial Autoencoder 2  " + PC.ENDC)
        print(PC.RED + " =========================================\n" + PC.ENDC)

    print(PC.BLUE + " network parameters  " + PC.ENDC)
    print(PC.BLUE + " ------------------\n" + PC.ENDC)
    print(model.get_parameters())

    return model


def init_parameters(param_file, model_type):
    if param_file is None:

        if model_type == 'VAE':

            parameters = {
                'latent_dim': 100,
                'layers_enc_dim': [1000, 500, 400, 300, 200],
                'layers_dec_dim': [200, 300, 400, 500, 1000],
                'batch_size': 1000,
                'epochs': 5000
            }

        elif model_type == 'AAE1':

            parameters = {
                'latent_dim': 100,
                'layers_enc_dim': [1000, 500, 400, 300, 200],
                'layers_dec_dim': [200, 300, 400, 500, 1000],
                'layers_dis_dim': [1000, 500, 400, 300, 200],
                'batch_size': 1000,
                'epochs': 100
            }

        elif model_type == 'AAE2':

            parameters = {
                'latent_dim': 100,
                'num_clusters': 17,
                'layers_enc_dim': [1000, 500, 400, 300, 200],
                'layers_dec_dim': [200, 300, 400, 500, 1000],
                'layers_dis_dim': [1000, 500, 400, 300, 200],
                'layers_dis_cat_dim': [1000, 500, 400, 300, 200],
                'batch_size': 1000,
                'epochs': 100,
                'tau': 0.05  # temperature parameter
            }

    else:

        file_settings = open(param_file, "r")
        with file_settings as f:
            parameters = json.load(f)

    return parameters


parser = argparse.ArgumentParser(
    prog='train.py',
    usage='python train.py --data-file=data_PATH [options]',
    description="The %(prog)s script allows comstruct and train a deepneural network for clustering of single-cell \
                 RNA sequencing data. The models implemented include yhe variational autoencoder and two different\
                 implementations of adversarial autoencoder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

requiredargs = parser.add_argument_group('required arguments')
optionalargs = parser.add_argument_group('options')

requiredargs.add_argument(
    "-f",
    "--data-file",
    type=str,
    dest="data_file",
    default=None,
    help="path to processed single cell RNA data file in h5da format."
)

requiredargs.add_argument(
    "-m",
    "--model",
    type=str,
    dest="model_type",
    default="VAE",
    help="select model type: 'VAE', 'AAE1', 'AAE2'."
)

optionalargs.add_argument(
    "-p",
    "--param",
    type=str,
    dest="param_file",
    default=None,
    help="JSON file with input parameters to initialize the model."
)

optionalargs.add_argument(
    "-ap",
    "--add-param",
    type=str,
    dest="add_param_file",
    default=None,
    help="JSON file with additional model training parameters."
)

optionalargs.add_argument(
    "-tf",
    "--tfrecord-data-file",
    type=str,
    dest="tfrecord_data_file",
    default=None,
    help="path to TFRecord dataset file."
)

optionalargs.add_argument(
    "-ds",
    "--distribution-strategy",
    type=str,
    dest="strategy_type",
    default='MirroredStrategy',
    help="TensorFlow distribution strategy: 'MirroredStrategy', 'MultiWorkerMirroredStrategy'."
)

optionalargs.add_argument(
    "-o",
    "--output",
    type=str,
    dest="output_folder",
    default='output',
    help="name of the output folder."
)

optionalargs.add_argument(
    "-t",
    "--task",
    type=str,
    dest="task",
    default=None,
    help="task number for multi-worker distribution strategy."
)


def main(data_file, model_type, strategy_type, param_file, add_param_file, tfrecord_data_file, task, output_folder):
    """

    :param data_file:
        path to processed single cell RNA data file in h5da format
    :param model_type:
        model type is one of 'VAE', 'AAE1', and 'AAE2'
    :param strategy_type:
        TensorFlow distribution strategy type: 'MirroredStrategy' or 'MultiWorkerMirroredStrategy'
    :param param_file:
        JSON file with input parameters to initialize the model
    :param add_param_file:
        JSON file with additional model training parameters
    :param tfrecord_data_file:
        path to TFRecord dataset file with extension .tfrecord
    :param task:
        task number for multi-worker distribution strategy
    :param output_folder:
        name of the output folder
    """

    tf.enable_eager_execution()
    tf.executing_eagerly()

    if data_file is None:
        print(PC.WARNING + "Specify path to h5da data file (-f / --data-file).\n" + PC.ENDC)
        sys.exit()

    header()

    if output_folder[-1] != "/":
        output_folder = output_folder + "/"

    # Initialize model
    model = init_model(param_file, model_type)

    print("\n")
    print(PC.BLUE + " load and transform the data  " + PC.ENDC)
    print(PC.BLUE + " ---------------------------\n" + PC.ENDC)
    # Load data
    model.load_data(data_file)

    # Rescale data
    model.rescale_data()

    print("\n")

    # Build and compile the model
    print(PC.BLUE + " build and compile model  " + PC.ENDC)
    print(PC.BLUE + " -----------------------\n" + PC.ENDC)

    # Set distribution strategy
    if strategy_type == 'MirroredStrategy':
        CROSS_DEVICE_OPS = tf.distribute.NcclAllReduce()
        # CROSS_DEVICE_OPS = tf.distribute.ReductionToOneDevice()
        # CROSS_DEVICE_OPS = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=CROSS_DEVICE_OPS)

        with strategy.scope():
            if model_type == 'VAE':
                BATCH_SIZE_PER_REPLICA = model.batch_size
                global_batch_size = (BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync)
                model.batch_size = global_batch_size
                model.build_model()

            else:
                model.build_model()

            model.get_summary()

    elif strategy_type == 'MultiWorkerMirroredStrategy':

        os.environ['TF_CONFIG'] = json.dumps({
            "cluster": {
                "worker": ["localhost:2222", "localhost:2223"]
            },
            "task": {"type": "worker", "index": task}
        })

        tf_config = os.getenv('TF_CONFIG')
        print("cluster configuration: {}\n".format(tf_config))

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

        with strategy.scope():
            if model_type == 'VAE':
                BATCH_SIZE_PER_REPLICA = model.batch_size
                global_batch_size = (BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync)
                model.batch_size = global_batch_size
                model.build_model()

            else:
                model.build_model()

            model.get_summary()

    else:
        print('Invalid distribution strategy')
        sys.exit()

    print("\n")

    # Export model graphs
    print(PC.BLUE + " export model graphs  " + PC.ENDC)
    print(PC.BLUE + " -------------------\n" + PC.ENDC)

    log_dir = '../results/' + model_type + '/' + output_folder
    os.makedirs(log_dir + 'graphs', exist_ok=True)
    model.export_graph(log_dir + 'graphs')

    # Training
    print("\n")
    print(PC.BLUE + " model training  " + PC.ENDC)
    print(PC.BLUE + " -------------- \n" + PC.ENDC)

    os.makedirs(log_dir + '/losses', exist_ok=True)

    if model_type == 'VAE':

        if add_param_file is None:
            additional_parameters = {'val_split': 0.2,
                                     'log_dir': log_dir,
                                     'num_workers': 24}
        else:
            additional_parameters = json.load(add_param_file)

        if tfrecord_data_file is None:
            loss, val_loss = model.train(mode='Dataset',
                                         **additional_parameters)

        else:
            loss, val_loss = model.train(mode='TFRecord',
                                         data_file=tfrecord_data_file,
                                         **additional_parameters)

        with open(log_dir + '/losses/loss.txt', 'w') as f:
            for k in loss:
                f.write("%s\n" % k)

        with open(log_dir + '/losses/val_loss.txt', 'w') as f:
            for k in val_loss:
                f.write("%s\n" % k)

        # Save training loss and validation loss to file
        print('Train loss and validation loss exported to file.')

    elif model_type == 'AAE1':
        if add_param_file is None:

            additional_parameters = {'enable_function': True,
                                     'graph': False,
                                     'gene': None}
        else:
            additional_parameters = json.load(add_param_file)

        BATCH_SIZE_PER_REPLICA = model.batch_size
        global_batch_size = (BATCH_SIZE_PER_REPLICA *
                             strategy.num_replicas_in_sync)

        # Create input dataset
        if tfrecord_data_file is None:
            train_dataset = tf.data.Dataset.from_tensor_slices(model.data).shuffle(
                len(model.data)).repeat(model.epochs).batch(global_batch_size, drop_remainder=True).prefetch(
                buffer_size=1)
        else:
            train_dataset = data_generator(tfrecord_data_file,
                                           batch_size=global_batch_size,
                                           epochs=model.epochs,
                                           is_training=True)

        with strategy.scope():
            distributed_train_dataset = strategy.experimental_distribute_dataset(train_dataset)

            rec_loss, dis_loss = model.distributed_train(distributed_train_dataset, strategy,
                                                         log_dir=log_dir,
                                                         **additional_parameters)

            with open(log_dir + '/losses/rec_loss.txt', 'w') as f:
                for k in rec_loss:
                    f.write("%s\n" % k)

            with open(log_dir + '/losses/dis_loss.txt', 'w') as f:
                for k in dis_loss:
                    f.write("%s\n" % k)

            # Save training loss and validation loss to file
            print('Reconstruction loss and discriminator loss exported to file.')

    elif model_type == 'AAE2':
        if add_param_file is None:

            additional_parameters = {'enable_function': True,
                                     'graph': False,
                                     'gene': None}
        else:
            additional_parameters = json.load(add_param_file)

        BATCH_SIZE_PER_REPLICA = model.batch_size
        global_batch_size = (BATCH_SIZE_PER_REPLICA *
                             strategy.num_replicas_in_sync)

        # Create input dataset
        if tfrecord_data_file is None:
            train_dataset = tf.data.Dataset.from_tensor_slices(model.data).shuffle(
                len(model.data)).repeat(model.epochs).batch(global_batch_size, drop_remainder=True).prefetch(
                buffer_size=1)
        else:
            train_dataset = data_generator(tfrecord_data_file,
                                           batch_size=global_batch_size,
                                           epochs=model.epochs,
                                           is_training=True)

        with strategy.scope():

            distributed_train_dataset = strategy.experimental_distribute_dataset(train_dataset)

            rec_loss, dis_loss, dis_cat_loss = model.distributed_train(distributed_train_dataset, strategy,
                                                                       log_dir=log_dir,
                                                                       **additional_parameters)

            with open(log_dir + '/losses/rec_loss.txt', 'w') as f:
                for k in rec_loss:
                    f.write("%s\n" % k)

            with open(log_dir + '/losses/dis_loss.txt', 'w') as f:
                for k in dis_loss:
                    f.write("%s\n" % k)

            with open(log_dir + '/losses/dis_cat_loss.txt', 'w') as f:
                for k in dis_cat_loss:
                    f.write("%s\n" % k)

            # Save training loss and validation loss to file
            print('Reconstruction loss, discriminator loss, and categorical discriminator loss exported to file.')


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(**vars(arguments))
