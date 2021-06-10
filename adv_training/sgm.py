import tensorflow as tf
from attacks.attack import Attack


class SGM(Attack):
    def __init__(self, out_dir, eps_init=0.001, nb_iter=1000, smoothness=0.1, mu=1.0, eps=0.1):
        super().__init__()

        self.mu = mu
        self.eps = eps
        self.eps_init = eps_init
        self.nb_iter = nb_iter
        self.smoothness = smoothness

        self.out_dir = out_dir
        self.f = None
        self.coef_shifted = None

    def _loss_function(self, x, r, y_target):
        x_r = x + r

        kl_loss = self.mu * tf.keras.backend.categorical_crossentropy(
            y_target, self.f(x_r))

        shifted_r = tf.roll(r, shift=-1, axis=1) * self.coef_shifted
        fused_lasso = self.smoothness * tf.reduce_sum(tf.abs(r - shifted_r), axis=1)

        l2_loss = tf.norm(r, axis=1)

        return tf.reduce_mean(kl_loss + l2_loss + fused_lasso)


    def perturb(self, model, input_x, input_y):
        self.f = self._model_as_tf_function(model)
        self.x_test = input_x
        self.y_test = input_y

        opt = tf.keras.optimizers.Adam()

        losses = np.zeros(shape=(self.nb_iter, 1))

        n = self.x_test.shape[0]
        self.p_x_test = self.x_test.copy()
        self.p_y_pred = np.zeros(shape=self.y_test.shape, dtype=np.float32)
        self.y_pred = np.zeros(shape=self.y_test.shape, dtype=np.float32)

        denom = 0
        # loop through batches for OOM issues
        for i in range(self.batch_size, n + self.batch_size, self.batch_size):
            denom += 1
            max_i = min(n, i)
            cur_i = i - self.batch_size

            cur_x_test = self.x_test[cur_i:max_i]
            x = tf.constant(cur_x_test, dtype=tf.float32)

            self.coef_shifted = np.ones(cur_x_test.shape)
            self.coef_shifted[:, -1, :] = 0
            self.coef_shifted = tf.constant(self.coef_shifted, dtype=tf.float32)

            # define constraint
            def constraint_eps(t):
                # t is the tensor
                return tf.clip_by_value(t, -self.eps, self.eps)

            # pick random initial perturbation
            r_data = np.random.randint(low=-1, high=1, size=x.shape) * self.eps_init
            r = tf.Variable(r_data, dtype=tf.float32, constraint=constraint_eps)

            y_target, y_pred = self._get_y_target(x)

            loss_func = lambda: self._loss_function(x, r, y_target)

            for itr in range(self.nb_iter):
                opt.minimize(loss_func, var_list=[r])
                losses[itr] = losses[itr] + self._loss_function(x, r, y_target)

            self.p_x_test[cur_i:max_i] = self.p_x_test[cur_i:max_i] + r.numpy()
            self.p_y_pred[cur_i:max_i] = self.f(tf.constant(self.p_x_test[cur_i:max_i],
                                                            dtype=tf.float32)).numpy()
            self.y_pred[cur_i:max_i] = y_pred

        return self.p_x_test
