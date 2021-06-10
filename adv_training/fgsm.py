import tensorflow as tf
from attacks.attack import Attack


class FGSM(Attack):
    def __init__(self, out_dir, eps=0.1):
        super().__init__()
        self.eps = eps
        self.out_dir = out_dir
        self.f = None

    def _loss_function(self, x, y_target):
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_target, self.f(x)))

    def perturb(self, model, input_x, input_y):
        self.f = self._model_as_tf_function(model)

        x = tf.Variable(input_x, dtype=tf.float32)

        y_target, _ = self._get_y_target(x)

        with tf.GradientTape(persistent=True) as t:
            loss_func = self._loss_function(x, y_target)

        var_list = [x]

        grad = t.gradient(loss_func, var_list)[0]

        grad = tf.sign(grad)
        grad = tf.stop_gradient(grad)

        # scale gradient (perturbation) to ensure eps neighborhood
        grad = -1.0 * grad * self.eps

        return input_x + grad.numpy()
