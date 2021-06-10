import os
import pandas as pd
import numpy as np

import tensorflow as tf
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Attack(object):

    def __init__(self):
        self.batch_size = 32
        self.p_x_test = None
        self.x_test = None
        self.y_test = None
        self.out_dir = None
        self.p_y_pred = None
        self.f = None
        self.y_pred = None

    @staticmethod
    def compute_snr(arr, axis=1):
        a = np.asanyarray(arr)
        m = a.mean(axis) ** 2
        varr = a.var(axis=axis)
        return np.mean(np.where(varr == 0, 0, m / varr))

    def _loss_function(self, **kwargs):
        """
        Defines the loss function that is minimized by the generated perturbation
        """
        error = "Sub-classes must implement generate."
        raise NotImplementedError(error)

    def _model_as_tf_function(self, model):
        """
        Defines the keras model as a tensorflow function.
        """
        new_input_layer = model.inputs
        new_output_layer = model.layers[-1].output
        new_feed_forward = tf.keras.models.Model(inputs=new_input_layer,
                                                 outputs=new_output_layer)

        @tf.function
        def f(x):
            return new_feed_forward(x)

        return f

    def _get_y_target(self, x, method='average_case'):
        """
        :param x: the input tensor
        :param method: how you want to generate the target
        :return: the target predictions, the predictions for the unperturbed data
        """
        if method == 'average_case':
            y_pred = self.f(x)
            y_pred = tf.stop_gradient(y_pred).numpy()

            target_p_logits = np.ones(shape=y_pred.shape) * 1e-5

            for i in range(len(y_pred)):
                c = y_pred[i].argmax(axis=0)
                c_s = list(range(y_pred.shape[1]))
                c_s.remove(c)
                new_c = np.random.choice(c_s)
                target_p_logits[i, new_c] = 1.0

            target_p_logits = tf.constant(target_p_logits, dtype=tf.float32)
            return tf.nn.softmax(target_p_logits), y_pred
        else:
            raise Exception('Chosen method not defined')

    def perturb(self, **kwargs):
        """
        Defines the perturbation aka the attack to be performed.
        Should return the compute_df_metrics(duration)
        """
        error = "Sub-classes must implement generate."
        raise NotImplementedError(error)

    def compute_df_metrics(self, duration):
        """
        Computes the original accuracy as well as the accuracy over the perturbed test set.
        Only correctly classifed test set samples are used for evaluation.
        """
        columns = ['asr', 'duration', 'count_true', 'count_false', 'avg_distance', 'avg_distance_true', 'avg_distance_false', 'snr']
        res = pd.DataFrame(data=np.zeros((1, len(columns)), dtype=np.float),
                           index=[0], columns=columns)

        # ground truth for evaluation
        y_true = self.y_pred.argmax(axis=1)
        # get one hot label
        p_y_true = self.p_y_pred.argmax(axis=1)
        
        # average distance
        res['avg_distance'] = tf.reduce_mean(tf.norm(self.p_x_test - self.x_test, axis=1)).numpy()

        # subset of correctly classified series
        idx = p_y_true == y_true
        count_true = idx.sum()
        res['count_true'] = count_true

        if count_true > 0:
            # Average distance between not perturbed (failed) and real
            s_p_x_test = self.p_x_test[idx]
            s_x_test = self.x_test[idx]
            
            res['avg_distance_true'] = tf.reduce_mean(tf.norm(s_p_x_test - s_x_test, axis=1)).numpy()
        else:
            res['avg_distance_true'] = np.nan

        # Average Success Rate (ASR)
        res['asr'] = 1.0 - accuracy_score(y_true, p_y_true)

        # Subset of successfully misclassified from the old subset
        idx = p_y_true != y_true

        count_false = idx.sum()
        res['count_false'] = count_false

        if count_false > 0:
            # Average distance between perturbed and real
            s_p_x_test = self.p_x_test[idx]
            s_x_test = self.x_test[idx]

            res['avg_distance_false'] = tf.reduce_mean(tf.norm(s_p_x_test - s_x_test, axis=1)).numpy()
            res['snr'] = self.compute_snr(self.p_x_test)
        else:
            res['avg_distance_false'] = np.nan
            res['snr'] = np.nan

        res['duration'] = duration
        return res

    def save_p_x_test(self):
        np.savetxt(os.path.join(self.out_dir, 'p_x_test.tsv'), self.p_x_test.squeeze(), delimiter='\t')
        np.save(os.path.join(self.out_dir, 'p_y_pred.npy'), self.p_y_pred)

    def save_loss(self, losses):
        np.savetxt(os.path.join(self.out_dir, 'losses.csv'), losses, delimiter=',')
        plt.figure()
        plt.title('Loss')
        plt.plot(losses, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.out_dir, 'losses.png'))
        plt.close()

    def plot(self, idx_to_plot=0):
        """
        Plots the corresponding original time series with its perturbation.
        """
        p_y_true = self.p_y_pred.argmax(axis=1)[idx_to_plot]
        y_true = self.y_pred.argmax(axis=1)[idx_to_plot]

        plt.figure()
        plt.plot(self.x_test[idx_to_plot], color='blue',
                 label='original-class-' + str(y_true))
        plt.plot(self.p_x_test[idx_to_plot], color='red',
                 label='perturbed-class-' + str(p_y_true))
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.out_dir, 'example-attack.png'))
        plt.close()

        # plot the perturbation
        perturbation = self.p_x_test[idx_to_plot] - self.x_test[idx_to_plot]
        plt.figure()
        #plt.title('Perturbation')
        plt.plot(perturbation)
        plt.savefig(os.path.join(self.out_dir, 'example-perturbation.pdf'))
        plt.close()
