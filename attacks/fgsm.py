import tensorflow as tf
import time
from attacks.attack import Attack
import numpy as np
import os

class FGSM(Attack):
    def __init__(self, model_dir, x_test, y_test, out_dir, eps=0.1):
        super().__init__()
        self.f = self._model_as_tf_function(tf.keras.models.load_model(model_dir))
        self.x_test = x_test
        self.y_test = y_test  # used for evaluation
        self.eps = eps
        self.out_dir = out_dir

        self.p_x_test = None
        self.p_y_pred = None
        self.y_pred = None

    def _loss_function(self, x, y_target):
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_target, self.f(x)))

    def perturb(self):
        start_time = time.time()

        n = self.x_test.shape[0]
        self.p_x_test = self.x_test.copy()
        self.p_y_pred = np.zeros(shape=self.y_test.shape, dtype=np.float32)
        self.y_pred = np.zeros(shape=self.y_test.shape, dtype=np.float32)

        # loop through batches for OOM issues
        for i in range(self.batch_size, n + self.batch_size, self.batch_size):
            max_i = min(n, i)
            cur_i = i - self.batch_size

            cur_x_test = self.x_test[cur_i:max_i]

            x = tf.Variable(cur_x_test, dtype=tf.float32)

            y_target, y_pred = self._get_y_target(x)

            with tf.GradientTape(persistent=True) as t:
                loss_func = self._loss_function(x, y_target)

            var_list = [x]

            grad = t.gradient(loss_func, var_list)[0]

            grad = tf.sign(grad)
            grad = tf.stop_gradient(grad)

            # scale gradient (perturbation) to ensure eps neighborhood
            grad = -1.0 * grad * self.eps

            self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + grad.numpy()
            self.p_y_pred[cur_i:max_i] = self.f(tf.constant(self.p_x_test[cur_i:max_i],
                                                            dtype=tf.float32)).numpy()
            self.y_pred[cur_i:max_i] = y_pred

        duration = time.time() - start_time

        df_metrics = self.compute_df_metrics(duration)

        df_metrics.to_csv(os.path.join(self.out_dir, 'df_metrics.csv'))

        self.plot()

        tf.keras.backend.clear_session()

        return df_metrics
