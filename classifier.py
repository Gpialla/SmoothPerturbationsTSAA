import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import time
import re
import pandas as pd
import os

from utils.utils import calculate_metrics
from utils.utils import save_test_duration
from utils.utils import shuffle
import matplotlib.pyplot as plt


class ScaleLayer(keras.layers.Layer):
    def __init__(self, scale_tf_variable):
        super(ScaleLayer, self).__init__()
        self.scale = scale_tf_variable
        self.init_scale_tf_variable = K.get_value(scale_tf_variable)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {'scale_tf_variable': self.init_scale_tf_variable}


class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = os.path.join(self.output_directory, 'best_model.hdf5')

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 16
		nb_epochs = 2000

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(os.path.join(self.output_directory, 'last_model.hdf5'))

		model = keras.models.load_model(os.path.join(self.output_directory, 'best_model.hdf5'))

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		#save_logs(self.output_directory, hist, y_pred, y_true, duration)
        # TODO: To fix save logs
		keras.backend.clear_session()

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = os.path.join(self.output_directory, 'best_model.hdf5')
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,
                 build=True, batch_size=64, lr=0.001, nb_filters=32, use_residual=True,
                 use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=750,
                 bottleneck_size=32, train_method='augment', split_batch_norm=False,
                 adv_training=True):

        self.output_directory = output_directory
        self.train_method = train_method
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.split_batch_norm = split_batch_norm
        self.verbose = verbose
        self.scale_tf_variable_adv = K.variable(0.0)
        self.scale_tf_variable_normal = K.variable(1.0)
        self.adv_training = adv_training

        if not self.adv_training:
            self.nb_epochs = 2 * self.nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
            
    def get_model(self):
        return self.model

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)
        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)

        x = self.apply_batch_norm(x)

        x = keras.layers.Activation(activation='relu')(x)
        return x

    def apply_batch_norm(self, input_tensor):
        if self.split_batch_norm:
            # then we should use two batchnorm layer
            out_1 = keras.layers.BatchNormalization()(input_tensor)
            out_1 = ScaleLayer(self.scale_tf_variable_normal)(out_1)
            out_2 = keras.layers.BatchNormalization()(input_tensor)
            out_2 = ScaleLayer(self.scale_tf_variable_adv)(out_2)
            out_ = keras.layers.Add()([out_1, out_2])
        else:
            out_ = keras.layers.BatchNormalization()(input_tensor)

        return out_

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)

        shortcut_y = self.apply_batch_norm(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def insert_layer_nonseq(self, layer_regex, insert_layer_factory,
                            insert_layer_name=None, position='replace'):
        # source: https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in self.model.layers:
            for node in layer.outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {self.model.layers[0].name: self.model.input})

        # Iterate over all layers after the input
        for layer in self.model.layers[1:]:

            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name):
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                new_layer = insert_layer_factory()
                # if insert_layer_name:
                #     new_layer.name = insert_layer_name
                # else:
                #     new_layer.name = '{}_{}'.format(layer.name,
                #                                     new_layer.name)

                x = new_layer(x)
                print('Layer {} inserted {} layer {}'.format(new_layer.name,
                                                             position, layer.name))
                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

        return keras.models.Model(inputs=self.model.inputs, outputs=x)

    def check_if_models_sharing_weights(self):
        res = 0
        for i in range(len(self.model.layers)):
            if isinstance(self.model.layers[i], keras.layers.BatchNormalization):
                continue
            if len(self.model_for_gan.layers[i].get_weights()) == 0:
                continue
            w1 = self.model_for_gan.layers[i].get_weights()[0]
            w2 = self.model.layers[i].get_weights()[0]
            res += (w1 - w2).sum()

        assert res == 0.0

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_test, y_test, perturber):
        start_time = time.time()

        n = x_train.shape[0]

        accs = []
        losses = []
        val_accs = []
        val_losses = []

        file_path = self.output_directory + 'best_model.hdf5'

        min_loss = np.inf

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
        reduce_lr.set_model(self.model)
        reduce_lr.on_train_begin()
        reduce_lr.verbose = self.verbose

        for e in range(self.nb_epochs):
            x_train, y_train = shuffle(x_train, y_train)

            if self.verbose:
                print('epoch', e)

            acc_ = 0
            loss_ = 0

            denom = 0

            for i in range(self.batch_size, n + self.batch_size, self.batch_size):
                max_i = min(n, i)
                cur_i = i - self.batch_size

                if self.verbose:
                    print('cur_i', cur_i)

                x = x_train[cur_i:max_i]
                y = y_train[cur_i:max_i]

                if not self.adv_training:
                    curr_loss_1, curr_acc_1 = self.model.train_on_batch(x, y)
                    curr_loss_2 = curr_loss_1
                    curr_acc_2 = curr_acc_1
                else:

                    new_x = perturber.perturb(self.model, x, y)

                    if self.split_batch_norm:
                        # train on perturbed
                        K.set_value(self.scale_tf_variable_normal, 0.0)
                        K.set_value(self.scale_tf_variable_adv, 1.0)
                        curr_loss_2, curr_acc_2 = self.model.train_on_batch(x, y)
                        # train normal unperturbed series
                        K.set_value(self.scale_tf_variable_normal, 1.0)
                        K.set_value(self.scale_tf_variable_adv, 0.0)
                        curr_loss_1, curr_acc_1 = self.model.train_on_batch(x, y)
                    else:
                        # train twice on the same batch to simulate the same number of batch updates
                        x = np.concatenate((new_x, x), axis=0)
                        y = np.concatenate((y.copy(), y.copy()))
                        x, y = shuffle(x, y)
                        nn = len(x)
                        curr_loss_1, curr_acc_1 = self.model.train_on_batch(x[:nn // 2], y[:nn // 2])
                        curr_loss_2, curr_acc_2 = self.model.train_on_batch(x[nn // 2:], y[nn // 2:])

                loss_ += curr_loss_1 + curr_loss_2
                acc_ += curr_acc_1 + curr_acc_2

                denom += 1

            val_loss_, val_acc_ = self.model.evaluate(x_test, y_test,
                                                      batch_size=self.batch_size, verbose=False)

            acc_ = acc_ / (2 * denom)
            loss_ = loss_ / (2 * denom)

            accs.append(acc_)
            val_accs.append(val_acc_)
            losses.append(loss_)
            val_losses.append(val_loss_)

            if loss_ < min_loss:
                min_loss = loss_
                self.model.save_weights(file_path)

            reduce_lr.on_epoch_end(epoch=e, logs={'loss': loss_})

        plt.figure()
        plt.ylim(top=1.0, bottom=0.0)
        plt.plot(accs, label='train', color='blue')
        plt.plot(val_accs, label='test', color='red')
        plt.legend(loc='best')
        plt.savefig(self.output_directory + 'acc.pdf')
        plt.close()

        plt.figure()
        plt.plot(losses, label='train', color='blue')
        plt.plot(val_losses, label='test', color='red')
        plt.legend(loc='best')
        plt.savefig(self.output_directory + 'loss.pdf')
        plt.close()

        df = pd.DataFrame(index=[i for i in range(self.nb_epochs)],
                          columns=['loss', 'acc', 'val_loss', 'val_acc'])
        df['loss'] = losses
        df['acc'] = accs
        df['val_loss'] = val_losses
        df['val_acc'] = val_accs

        df.to_csv(self.output_directory + 'history.csv')

        duration = time.time() - start_time

        self.model.load_weights(file_path)

        y_pred = self.model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        res_df = calculate_metrics(y_true, y_pred, duration)
        res_df.to_csv(self.output_directory + 'df_metrics.csv', index=False)
        keras.backend.clear_session()
        return res_df

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
