# Imports
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('agg')
import numpy as np

from attacks.attack import Attack
import utils
from utils.utils import read_all_datasets
from utils.utils import prepare_data
from utils.utils import create_directory
from utils.utils import get_k_samples_from_class, save_k_dist
from utils.utils import plot_pairwise, plot_cm
from utils.utils import plot_double_pairwise
from utils.utils import get_attack_from_name_for_adv_training
from utils.utils import perturber_init
from utils.utils import check_if_file_exits
from utils.utils import split_nb_example_per_class
from utils.utils import attack_init
from utils.utils import get_attack_from_name
from utils.constants import ATTACK_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import UNIVARIATE_ARCHIVE_NAMES
from utils.constants import NB_ITERATIONS
from classifier import Classifier_INCEPTION


# Required on some devices
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
"""

# Environement variables
classifier_name = CLASSIFIERS[0]
archive_name = UNIVARIATE_ARCHIVE_NAMES[0]
root_dir = '/media/gautier/Data1/Datasets/'
datasets_dict = read_all_datasets(root_dir, archive_name)
nb_iterations = NB_ITERATIONS

action = sys.argv[1]

def run_adv_attacks():
    """Run the adversarial attacks over all datasets for the specified attack.

    Keyword arguments:
    attack_name -- The name of the adversarial attack
    """

    for itr_ in range(nb_iterations):
            print('iteration', itr_)
            for attack_name in ATTACK_NAMES:

                print('\tattack_name', attack_name)

                root_model_dir = os.path.join(root_dir, 'results', classifier_name, archive_name)
                root_out_dir = os.path.join(root_dir, 'adv', 'attack', attack_name, classifier_name, archive_name, 'itr_' + str(itr_))

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                    print('\t\tdataset_name: ', dataset_name)

                    _, _, x_test, y_test, _, _, _, _ = \
                        prepare_data(datasets_dict, dataset_name)

                    output_directory = os.path.join(root_out_dir, dataset_name)

                    test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')
                    test_dir_doing = os.path.join(output_directory, 'doing')

                    if check_if_file_exits(test_dir_df_metrics):
                        print('Already_done', root_out_dir, dataset_name)
                        continue

                    create_directory(test_dir_doing)

                    model_dir = os.path.join(root_model_dir, dataset_name, 'best_model.hdf5')

                    attack = attack_init(attack_name, model_dir, x_test, y_test, output_directory)

                    attack.perturb()

                    attack.save_p_x_test()

                    # the creation of this directory means
                    create_directory(os.path.join(output_directory, 'DONE'))

                    print('\t\t\tDONE')

def run_adv_training(attack_name):
    """Run the adversarial attacks over all datasets for the specified attack.

    Keyword arguments:
    attack_name -- The name of the adversarial attack
    """

    # Check if valid parameter
    if not attack_name in ATTACK_NAMES:
        print("Error %s is not an implemented adversarial attack!" % attack_name)
        exit()

    split_batch_norms = [False]
    nb_example_per_class = 5  # minimum number of training examples per class

    for itr_ in range(nb_iterations):
        trr = ''
        if itr_ != 0:
            trr = '_itr_' + str(itr_)

        for split_batch_norm in split_batch_norms:

            tmp_output_directory = os.path.join(root_dir, 'adv', 'adv_training', classifier_name, 'nb_prototype', str(nb_example_per_class), 'split_batch_norm', str(split_batch_norm), attack_name, archive_name, trr)

            for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                print('dataset_name: ', dataset_name)

                x_train, y_train, x_test, y_test, _, nb_classes, _, enc = prepare_data(datasets_dict, dataset_name)

                output_directory = os.path.join(tmp_output_directory, dataset_name)

                perturber = perturber_init(attack_name, output_directory)

                test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')
                test_dir_doing = os.path.join(output_directory, 'doing')

                test_dir_doing = create_directory(test_dir_doing)

                if check_if_file_exits(test_dir_df_metrics):
                    print('Already_done', output_directory)
                    continue
                elif test_dir_doing is None:
                    print('Already doing', tmp_output_directory, dataset_name)
                    continue

                create_directory(test_dir_doing)

                x_train, y_train = split_nb_example_per_class(
                    enc, nb_example_per_class, x_train, y_train)

                input_shape = x_train.shape[1:]

                classifier = Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                  split_batch_norm=split_batch_norm,
                                                  adv_training=True)

                classifier.fit(x_train, y_train, x_test, y_test, perturber)

                # the creation of this directory means it is finished
                create_directory(os.path.join(output_directory, 'DONE'))

def plot_pairwise(attack_name_1, attack_name_2, metric):
    """Makes a pairwise plot using the two specified methods.
    /!\ run_adv_attacks must have been run previously
    Metric can be either "asr" or "avg_distance" for the L2 norm

    Keyword arguments:
    attack_name_1 -- The name of the first attack
    attack_name_2 -- The name of the second attack
    metric        -- The metric used for the pairwise plot
    """

    output_dir = os.path.join(root_dir, 'adv', 'attack')
    plot_pairwise(output_dir, attack_name_1, attack_name_2, metric=metric)



if action == 'attack':
    run_adv_attacks()
elif action == "plot_pairwise":
    plot_pairwise(sys.argv[2], sys.argv[3], sys.argv[4])
elif action == "adv_training":
    run_adv_training(sys.argv[2])
elif action == 'run_all':
    print("Starting adversarial attacks")
    run_adv_attacks()

    print("Ploting all pairwise plots")
    plot_pairwise('bim', 'gm', "asr")
    plot_pairwise('bim', 'gm', "avg_distance")
    plot_pairwise('bim', 'gm-wo-clip', "asr")
    plot_pairwise('bim', 'gm-wo-clip', "avg_distance")
    plot_pairwise('bim', 'sgm-wo-clip', "asr")
    plot_pairwise('bim', 'sgm-wo-clip', "avg_distance")
    plot_pairwise('gm', 'gm-wo-clip', "asr")
    plot_pairwise('gm', 'gm-wo-clip', "avg_distance")
    plot_pairwise('gm', 'sgm-wo-clip', "asr")
    plot_pairwise('gm', 'sgm-wo-clip', "avg_distance")
    plot_pairwise('gm-wo-clip', 'sgm-wo-clip', "asr")
    plot_pairwise('gm-wo-clip', 'sgm-wo-clip', "avg_distance")

    print("Run SGM adversarial training")
    run_adv_training("sgm-wo-clip")
