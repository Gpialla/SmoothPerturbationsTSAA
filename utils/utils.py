import os
import operator
import utils

import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn

from scipy.stats import wilcoxon
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils.constants import ATTACK_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import NB_ITERATIONS
from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def normalize_norm(v, eps=None, avoid_zero_div=1e-12):
    axis = list(range(1, len(v.shape)))
    square = tf.maximum(avoid_zero_div,
                        tf.reduce_sum(tf.square(v), axis=axis, keepdims=True))
    if eps is None:
        # then normalize to a norm equal to one
        return tf.divide(v, tf.sqrt(square))
    else:
        # then normalize to ensure eps neighborhood
        factor = tf.minimum(1.0, tf.divide(eps, tf.sqrt(square)))
        return tf.multiply(factor, v)


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """

    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test


def prepare_data(datasets_dict, dataset_name):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    label_enc = sklearn.preprocessing.LabelEncoder()
    y_train = label_enc.fit_transform(y_train)
    y_test = label_enc.transform(y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')

    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc


def read_all_datasets(root_dir, archive_name):
    datasets_dict = {}

    if archive_name == 'UCRArchive_2018':

        for dataset_name in DATASET_NAMES_2018:
            root_dir_dataset = os.path.join(root_dir, archive_name, dataset_name)

            df_train = pd.read_csv(os.path.join(root_dir_dataset, dataset_name+'_TRAIN.tsv'), sep='\t', header=None)
            df_test  = pd.read_csv(os.path.join(root_dir_dataset, dataset_name+'_TEST.tsv'),  sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns  = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(),
                                           x_test.copy(), y_test.copy())

    elif archive_name == 'TSC':
        dataset_names_to_sort = []

        for dataset_name in DATASET_NAMES:
            root_dir_dataset = os.path.join(root_dir, 'archives', archive_name, dataset_name)
            file_name = os.path.join(root_dir_dataset, dataset_name)

            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict


def readucr(filename, delimiter=','):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def split_nb_example_per_class(enc, nb_example_per_class, x, y):
    if nb_example_per_class is None:
        return x, y
    y_arg_max = y.argmax(axis=1)
    classes = np.unique(y_arg_max)
    new_x = []
    new_y = []
    for c in classes:
        c_x = x[np.where(y_arg_max == c)]
        limit_np_example_per_class = min(nb_example_per_class, len(c_x))
        c_x = c_x[np.random.permutation(len(c_x))]
        c_x = c_x[:limit_np_example_per_class]
        new_x.extend(c_x.tolist())
        new_y.extend([c for i in range(limit_np_example_per_class)])

    new_x = np.array(new_x)
    new_y = enc.transform(np.array(new_y).reshape(-1, 1)).toarray()
    return new_x, new_y


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def shuffle(x_train, y_train):
    n = len(x_train)
    idx_train = np.random.permutation(n)
    return x_train[idx_train], y_train[idx_train]


def get_attack_from_name(attack_name):
    if attack_name == 'fgsm':
        from attacks.fgsm import FGSM
        return FGSM
    if attack_name == 'bim':
        from attacks.bim import BIM
        return BIM
    if attack_name == 'sgm':
        from attacks.sgm import SGM
        return SGM

def attack_init(attack_name, model_dir, x_test, y_test, output_directory):
    print("\n" + attack_name)
    if attack_name == 'sgm':
        from attacks.sgm import SGM
        return SGM(model_dir, x_test, y_test, output_directory)

    elif attack_name == 'sgm-wo-const':
        from attacks.sgm import SGM
        return SGM(model_dir, x_test, y_test, output_directory, eps=999)

    else:
        attack = get_attack_from_name(attack_name)
        return attack(model_dir, x_test, y_test, output_directory)

def get_attack_from_name_for_adv_training(attack_name):
    if attack_name == 'fgsm':
        from adv_training.fgsm import FGSM
        return FGSM

def perturber_init(attack_name, output_directory):
    from adv_training.fgsm import FGSM
    if attack_name == 'fgsm':
        return FGSM(output_directory, eps=0.1)
    elif attack_name == 'fgsm-wo-const':
        return FGSM(output_directory, eps=999)

def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_results_csv_attack(root_dir_results, metric='asr'):
    not_completed = set()
    columns = ['classifier_name', 'archive_name', 'dataset_name', 'itr',
               'attack_name', 'asr', 'distance', 'duration']
    res = pd.DataFrame(data=np.zeros((0, len(columns)), dtype=np.float), index=[],
                       columns=columns)

    for attack_name in ATTACK_NAMES:
        for classifier_name in CLASSIFIERS:
            for archive_name in ARCHIVE_NAMES:
                for iteration in range(NB_ITERATIONS):
                    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                        
                        df_metrics_dir = os.path.join(root_dir_results, attack_name, classifier_name, archive_name, 'itr_' + str(iteration), dataset_name, 'df_metrics.csv')

                        if check_if_file_exits(df_metrics_dir):
                            df_metrics = pd.read_csv(df_metrics_dir)
                            df_metrics['classifier_name'] = classifier_name
                            df_metrics['archive_name'] = archive_name
                            df_metrics['dataset_name'] = dataset_name
                            df_metrics['attack_name'] = attack_name
                            df_metrics['itr'] = iteration

                            res = pd.concat([res, df_metrics], axis=0, sort=False)
                        else:
                            print("Does not exists", dataset_name, attack_name, df_metrics_dir)
                            not_completed.add(dataset_name)

    res.to_csv(os.path.join(root_dir_results, 'results.csv'), index=False)

    res = pd.DataFrame({
        metric: res.groupby(
            ['attack_name', 'archive_name', 'dataset_name'])[metric].mean()
    }).reset_index()

    return res, not_completed

def plot_pairwise(root_dir, method_1, method_2, metric='asr'):
    print("Pairwise %s vs %s" % (method_1, method_2))
    print("- metric: %s\n" %metric)
    plt.figure()

    res_df, not_completed = read_results_csv_attack(root_dir, metric=metric)

    sorted_df = res_df.loc[(res_df['attack_name'] == method_1) | \
                           (res_df['attack_name'] == method_2)]. \
        sort_values(['attack_name', 'archive_name', 'dataset_name'])

    sorted_df.reset_index(inplace=True)
    sorted_df.drop('index', axis=1, inplace=True)

    # remove non completed
    for not_completed_dname in not_completed:
        sorted_df.drop(sorted_df.loc[sorted_df['dataset_name'] == not_completed_dname].index,
                       axis=0, inplace=True)
    sorted_df.reset_index(inplace=True)
    sorted_df.drop('index', axis=1, inplace=True)

    # number of classifier we are comparing is 2 since pairwise
    m = 2

    max_nb_datasets = sorted_df.shape[0] // 2

    data = np.array(sorted_df[metric]).reshape(m, max_nb_datasets).transpose()

    # concat the dataset name and the archive name to put them in the columns s
    sorted_df['archive_dataset_name'] = sorted_df['archive_name'] + '__' + \
                                        sorted_df['dataset_name']

    # create the data frame containg the accuracies
    df_data = pd.DataFrame(data=data, columns=np.sort([method_1, method_2]))
                           #index=np.unique(sorted_df['archive_dataset_name']))

    #df_data.to_csv("/media/gautier/Data1/test.csv")

    x = df_data[method_1]
    y = df_data[method_2]

    plt.scatter(x=x, y=y, color='blue')

    _step = 0.001
    _min = min(x.min(), y.min()) - _step
    _max = max(x.max(), y.max()) + _step
    plt.xlim(_min, _max)
    plt.ylim(_min, _max)
    xx = np.arange(start=_min, stop=_max, step=_step)
    plt.plot(xx, xx, color='black')

    uniq, counts = np.unique(df_data[method_1] < df_data[method_2], return_counts=True)
    wins = counts[-1]
    uniq, counts = np.unique(df_data[method_1].equals(df_data[method_2]), return_counts=True)
    draws = counts[-1]
    uniq, counts = np.unique(df_data[method_1] > df_data[method_2], return_counts=True)
    losses = counts[-1]

    plt.xlabel("{}\n({} {}, draws {}, {} {})".format(
        method_1,
        method_1, wins,
        draws,
        method_2, losses 
        ), fontsize='large')
    plt.ylabel(method_2, fontsize='large')
    if metric == "avg_distance":
        metric = "L2 norm"
    plt.title(metric)

    print('Wins  :', wins, " (%s)" % method_2)
    print('Draws :', draws)
    print('Losses:', losses, " (%s)" % method_1)

    p_value = wilcoxon(df_data[method_1], df_data[method_2], zero_method='pratt')[1]
    print(p_value)

    name_plot = metric + '-' + method_1 + '-vs-' + method_2 + '-plot-pairwise.pdf'
    plt.savefig(os.path.join(root_dir, name_plot), bbox_inches='tight')

def plot_double_pairwise(root_dir, method_1, method_2, metrics=['asr', 'avg_distance']):
    print("Pairwise %s vs %s" % (method_1, method_2))
    print("- metrics: {}\n".format(metrics))

    n_metrics = len(metrics)
    fig, axs = plt.subplots(1, n_metrics, figsize=(7,4))

    for i_metric in range(n_metrics):
        res_df, not_completed = read_results_csv_attack(root_dir, metric=metrics[i_metric])

        sorted_df = res_df.loc[(res_df['attack_name'] == method_1) | (res_df['attack_name'] == method_2)]\
            .sort_values(['attack_name', 'archive_name', 'dataset_name'])

        sorted_df.reset_index(inplace=True)
        sorted_df.drop('index', axis=1, inplace=True)

        # remove non completed
        for not_completed_dname in not_completed:
            sorted_df.drop(sorted_df.loc[sorted_df['dataset_name'] == not_completed_dname].index,
                        axis=0, inplace=True)
        sorted_df.reset_index(inplace=True)
        sorted_df.drop('index', axis=1, inplace=True)

        # number of classifier we are comparing is 2 since pairwise
        m = 2

        max_nb_datasets = sorted_df.shape[0] // 2

        data = np.array(sorted_df[metrics[i_metric]]).reshape(m, max_nb_datasets).transpose()

        # concat the dataset name and the archive name to put them in the columns s
        sorted_df['archive_dataset_name'] = sorted_df['archive_name'] + '__' + \
                                            sorted_df['dataset_name']

        # create the data frame containg the accuracies
        df_data = pd.DataFrame(data=data, columns=np.sort([method_1, method_2]))
                            #index=np.unique(sorted_df['archive_dataset_name']))

        #df_data.to_csv("/media/gautier/Data1/test.csv")

        x = df_data[method_1]
        y = df_data[method_2]

        axs[i_metric].scatter(x=x, y=y, s=15, color='blue')

        _step = 0.001
        _min = min(x.min(), y.min()) - _step
        _max = max(x.max(), y.max()) + _step

        axs[i_metric].set_xlim(_min, _max)
        axs[i_metric].set_ylim(_min, _max)
        xx = np.arange(start=_min, stop=_max, step=_step)
        axs[i_metric].plot(xx, xx, color='black')

        wins = (df_data[method_1] > df_data[method_2]).sum()
        draws = (df_data[method_1] == df_data[method_2]).sum()
        losses = (df_data[method_1] < df_data[method_2]).sum()

        axs[i_metric].set_xlabel("{}\n({} {}, draws {}, {} {})".format(
            method_1,
            get_short_method_name(method_1), wins,
            draws,
            get_short_method_name(method_2), losses 
            ), fontsize='large')
        axs[i_metric].set_ylabel(method_2, fontsize='large')
        if metrics[i_metric] == "avg_distance":
            metrics[i_metric] = "L2 norm"
        axs[i_metric].set_title(metrics[i_metric])

        print("Metric %s" % metrics[i_metric])
        print('Wins  :', wins, " (%s)" % method_1)
        print('Draws :', draws)
        print('Losses:', losses, " (%s)" % method_2)

        p_value = wilcoxon(df_data[method_1], df_data[method_2], zero_method='pratt')[1]
        print(p_value)

    
    name_plot = method_1 + '-vs-' + method_2 + '-double-pairwise.pdf'
    save_path = os.path.join(root_dir, name_plot)
    print(save_path)
    plt.tight_layout()
    plt.savefig(save_path)

def get_short_method_name(name):
    if name == "gm-wo-clip":
        return "gm"
    if name == "sgm-wo-clip":
        return "sgm"
    return name

def plot_compare_perturbed_series(root_dir, method_1, method_2, iteration):
    root_dir_attack = os.path.join(root_dir, 'adv', 'attack')
    classifier_name = CLASSIFIERS[0]
    archive_name = ARCHIVE_NAMES[0]

    dataset_dicts = read_all_datasets(root_dir, archive_name)
    dnames = utils.constants.dataset_names_for_archive[archive_name]

    n = len(dnames)

    denom = 3
    fig, axs = plt.subplots(n // denom, denom, figsize=(7,4))

    i = 0
    j = 0
    for dataset_name in dnames:
        p_x_test_dir_1 = os.path.join(root_dir_attack, method_1, classifier_name, archive_name, 'itr_' + str(iteration), dataset_name, 'p_x_test.tsv')
        p_x_test_1 = pd.read_csv(p_x_test_dir_1, sep='\t', header=None).values

        p_x_test_dir_2 = os.path.join(root_dir_attack, method_2, classifier_name, archive_name, 'itr_' + str(iteration), dataset_name, 'p_x_test.tsv')
        p_x_test_2 = pd.read_csv(p_x_test_dir_2, sep='\t', header=None).values

        x_test = dataset_dicts[dataset_name][2]

        idx = np.random.choice(len(x_test))

        linewidth = 0.7

        # axs[i, j].plot(x_test[idx], color='gray', label='original')
        axs[i, j].plot(p_x_test_1[idx] - x_test[idx], color='blue', label=method_1, linewidth=linewidth)
        axs[i, j].plot(p_x_test_2[idx] - x_test[idx], color='red', label=method_2, linewidth=linewidth)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

        if i == 0 and j == 0:
            axs[i, j].legend(loc='best', fontsize='xx-small')

        axs[i, j].set_title(dataset_name)

        if i < n // denom - 1:
            i += 1
        else:
            j += 1
            i = 0

    name_plot = 'plt_compare-' + method_1 + '-vs-' + method_2 + '.pdf'
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir_attack, name_plot))


def get_k_samples_from_class(k, _class, y):
    idx_k = np.where(y == _class)[0]
    return np.random.choice(idx_k, size=k, replace=False)

def plot_cm(a, out_dir):
    # Plot confusion matrix of classes (original vs perturbed)
    df_cm = pd.DataFrame(
        a, 
        index   = [str(i) for i in range(a.shape[0])],
        columns = [str(i) for i in range(a.shape[1])]
    )
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix", fontsize=16)
    plt.xlabel("Class perturbed", fontsize=14)
    plt.ylabel("Class of origin", fontsize=14)
    plt.savefig(os.path.join(out_dir, 'cm-class.png'))
    plt.close()

def save_k_dist(k, nb_class, cm_class, origin_dist, perturb_dist, output_dir):

    df_dist = pd.DataFrame(columns=["class_origin", "class_perturb", "n_samples", "avg_dist_class_origin", "avg_dist_class_perturb"])

    for i_origin in range(nb_class):
        for i_perturb in range(nb_class):
            
            n_samples = cm_class[i_origin, i_perturb]

            if n_samples == 0:
                continue
            
            avg_origin  = origin_dist[i_origin, i_perturb] / n_samples
            avg_perturb = perturb_dist[i_origin, i_perturb] / n_samples
        
            df_dist = df_dist.append({
                "class_origin": i_origin,
                "class_perturb": i_perturb,
                "n_samples": n_samples,
                "avg_dist_class_origin": avg_origin,
                "avg_dist_class_perturb": avg_perturb
            }, ignore_index=True)
    
    df_dist.to_csv(os.path.join(output_dir, "avg_{}_dist.csv".format(k)))