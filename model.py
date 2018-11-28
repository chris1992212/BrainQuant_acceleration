import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import scipy.io as sio
from tqdm import tqdm
import numpy as np
import time
import os
import utils
import h5py
class BrainQuantAI_Part_one(object):
    def __init__(self,
                 sess,
                 image_size_FE,
                 image_size_PE,
                 label_size,
                 is_train,
                 batch_size,
                 c_dim,
                 test_FE,
                 test_PE):
        self.sess = sess
        self.image_size_FE = image_size_FE
        self.image_size_PE = image_size_PE
        self.is_train = is_train
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.test_FE = test_FE
        self.test_PE = test_PE
        self.build_model()

    def log_record(self,config):
        log_dir = "log_{}".format('BrainQuantAI_Part_one')
        tl.files.exists_or_mkdir(log_dir)
        self.log_all, self.log_all_filename = utils.logging_setup(log_dir)
        utils.log_config(self.log_all_filename, config)

    def model(self, images, is_train = False, reuse = False):
        n_out = self.c_dim
        x = images
        _, nx, ny, nz = x.get_shape().as_list()

        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1, 0.02)

        with tf.variable_scope("u_net", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            inputs = tl.layers.InputLayer(x, name='inputs')

            conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1_1')
            conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1_2')
            conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                                   name='bn1')
            pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')


            conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2_1')
            conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2_2')
            conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                                   name='bn2')
            pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')



            conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3_1')
            conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3_2')
            conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                                   name='bn3')
            pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')


            conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv4_1')
            conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv4_2')
            conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                                   name='bn4')
            pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')


            conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv5_1')
            conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv5_2')
            conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init,
                                   name='bn5')


            print(" * After conv: %s" % conv5.outputs)

            up4 = tl.layers.DeConv2d(conv5, 512, (3, 3),
                                     out_size=[tf.to_int32(tf.shape(x)[1] / 8), tf.to_int32(tf.shape(x)[2] / 8)],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv4')
            up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn4_1')
            up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
            conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv4_1')
            conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv4_2')
            conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn4_2')


            up3 = tl.layers.DeConv2d(conv4, 256, (3, 3),
                                     out_size=[tf.to_int32(tf.shape(x)[1] / 4), tf.to_int32(tf.shape(x)[2] / 4)],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv3')
            up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn3_1')
            up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
            conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv3_1')
            conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv3_2')
            conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn3_2')


            up2 = tl.layers.DeConv2d(conv3, 128, (3, 3),
                                     out_size=[tf.to_int32(tf.shape(x)[1] / 2), tf.to_int32(tf.shape(x)[2] / 2)],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv2')
            up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn2_1')
            up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
            conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv2_1')
            conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv2_2')
            conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn2_2')


            up1 = tl.layers.DeConv2d(conv2, 64, (3, 3),
                                     out_size=[tf.to_int32(tf.shape(x)[1]), tf.to_int32(tf.shape(x)[2])],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv1')
            up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn1_1')
            up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
            conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv1_1')
            conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv1_2')
            conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn1_2')

            conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')

            out = tf.add(conv1.outputs, inputs.outputs, name='output')
            # input = inputs.outputs
            ######## -------------------------Data fidelity--------------------------------##########
            # for contrast in range(n_out):
            #     k_conv3 = utils.Fourier(conv1[:,:,:,contrast], separate_complex=False)
            #     mask = np.ones((batch_size, nx, ny))
            #     mask[:,:, 1:ny:3] = 0
            #     mask = np.fft.ifftshift(mask)
            #     # convert to complex tf tensor
            #     DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
            #     DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
            #     k_patches = utils.Fourier(input[:,:,:,contrast], separate_complex=False)
            #     k_space = k_conv3 * DEFAULT_MAKS_TF_c + k_patches*(1-DEFAULT_MAKS_TF_c)
            #     out = tf.ifft2d(k_space)
            #     out = tf.abs(out)
            #     out = tf.reshape(out, [batch_size, nx, ny, 1])
            #     if contrast == 0 :
            #         final_output = out
            #     else:
            #         final_output = tf.concat([final_output,out],3)
            ########-------------------------end------------------------------------###########3
            # print(" * Output: %s" % conv1.outputs)
            # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
            return out
    def model_multi_contrast(self,images,reuse = False):
            conv_in = list()
            block_one = utils.inference_block(images,images,reuse = reuse, name = 'block_one')
            conv_in.append(block_one)
            block_two = utils.inference_block(block_one, images, reuse=reuse, name='block_two')
            conv_in.append(block_two)
            x = utils.Concatenation(conv_in, name='concat1')
            block_three = utils.inference_block(x, images, reuse=reuse, name='block_three')
            conv_in.append(block_three)
            x = utils.Concatenation(conv_in, name='concat2' )
            block_four = utils.inference_block(x, images, reuse=reuse, name='block_four')
            conv_in.append(block_four)
            x = utils.Concatenation(conv_in, name='concat3')
            block_five = utils.inference_block(x, images, reuse=reuse, name='block_five')

            return block_five



    def data_input(self,config,testing_filename,shuffle = False,Batch_size =1):
        image_shape = (config.test_FE, config.test_PE, self.c_dim)
        y_image_shape = (config.test_FE, config.test_PE, self.c_dim)

        temp = h5py.File(testing_filename + '/low_CompI_final.mat')
        X_data = temp['low_CompI_final'].value
        X_data = np.transpose(X_data, [3, 2, 1, 0])  ###nPE,nSL,nFE*9,nCH

        temp = h5py.File(testing_filename + '/CompI_final.mat')
        Y_data = temp['CompI_final'].value
        Y_data = np.transpose(Y_data, [3, 2, 1, 0])
        nb_images = Y_data.shape[2]

        index_generator = utils._index_generator(nb_images, Batch_size, shuffle, None)

        while 1:
            index_array, current_index, current_batch_size = next(index_generator)

            batch_x = np.zeros((current_batch_size,) + image_shape)

            batch_y = np.zeros((current_batch_size,) + y_image_shape)

            for i, j in enumerate(index_array):


                batch_y[i, :, :, 0] = Y_data[:,:,j,0].astype('float32')
                batch_y[i, :, :, 1] = Y_data[:,:,j,1].astype('float32')
                batch_y[i, :, :, 2] = Y_data[:,:,j,2].astype('float32')
                batch_y[i, :, :, 3] = Y_data[:,:,j,3].astype('float32')
                batch_y[i, :, :, 4] = Y_data[:,:,j,4].astype('float32')
                batch_y[i, :, :, 5] = Y_data[:,:,j,5].astype('float32')

                batch_x[i, :, :, 0] = X_data[:,:,j,0].astype('float32')
                batch_x[i, :, :, 1] = X_data[:,:,j,1].astype('float32')
                batch_x[i, :, :, 2] = X_data[:,:,j,2].astype('float32')
                batch_x[i, :, :, 3] = X_data[:,:,j,3].astype('float32')
                batch_x[i, :, :, 4] = X_data[:,:,j,4].astype('float32')
                batch_x[i, :, :, 5] = X_data[:,:,j,5].astype('float32')
            yield (batch_x, batch_y)
    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size_FE, self.image_size_PE, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.image_size_FE, self.image_size_PE, self.c_dim], name='labels')
        self.validation_images = tf.placeholder(tf.float32, [None, self.test_FE, self.test_PE, self.c_dim], name='validation_images')
        self.validation_labels = tf.placeholder(tf.float32, [None, self.test_FE, self.test_PE, self.c_dim], name='validation_labels')

        self.pred = self.model(self.images, is_train = True, reuse = False)
        # self.pred = self.model_multi_contrast(self.images, reuse = False)

        self.preding_loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.srcing_loss = tf.reduce_mean(tf.square(self.labels - self.images))
        self.validation_pred = self.model(self.validation_images, is_train = False, reuse= True)
        # self.validation_pred = self.model_multi_contrast(self.validation_images, reuse= True)

        self.validation_preding_loss = tf.reduce_mean(tf.square(self.validation_labels - self.validation_pred))
        self.validation_srcing_loss = tf.reduce_mean(tf.square(self.validation_labels - self.validation_images))
        self.saver = tf.train.Saver()

    def train(self, config):
        self.log_record(config)
        test_data = self.data_input(config,config.testing_filename)
        training_data = self.data_input(config,config.training_filename)
        train_nx, train_ny = utils.tfrecord_read(config,self.c_dim)
        self.merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("tmp/")
        summary_writer.add_graph(self.sess.graph)

        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.preding_loss)
        tf.global_variables_initializer().run()
        # self.saver.restore(self.sess, config.save_model_filename)
        self.coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        # Train
        if config.is_train:
            print("Now Start Training...")
            best_mse = 1
            early_stop_number =40
            for epoch in tqdm(range(config.epoch)):
                # Run by batch images
                batch_xs, batch_ys = self.sess.run([train_nx, train_ny])
                # batch_xs, batch_ys = next(test_data)
                _, err, out, summary = self.sess.run([self.train_op, self.preding_loss, self.pred,self.merged_summary_op], feed_dict = {self.images: batch_xs, self.labels: batch_ys})

                summary_writer.add_summary(summary, epoch)
                if epoch % 100 == 0:
                    print('epoch %d training_cost => %.7f ' % (epoch, err))
                    save_path = self.saver.save (self.sess, config.save_model_filename)
                if epoch % 1000 == 0:
                    self.saver.restore(self.sess, config.save_model_filename)
                    test_src_loss = np.zeros((config.TESTING_NUM, 1))
                    test_pred_loss = np.zeros((config.TESTING_NUM, 1))
                    train_src_loss = np.zeros((config.TESTING_NUM, 1))
                    train_pred_loss = np.zeros((config.TESTING_NUM, 1))
                    for ep in range(config.TESTING_NUM):
                        batch_xs_validation, batch_ys_validation = next(test_data)
                        test_src_loss[ep], test_pred_loss[ep] = self.sess.run([self.validation_srcing_loss, self.validation_preding_loss],
                                                                       feed_dict = { self.validation_images: batch_xs_validation, self.validation_labels: batch_ys_validation})
                    print( 'epoch: %d ,ave_src_MSE: %.7f,ave_pred_MSE: %.7f' % ( epoch, test_src_loss.mean(), test_pred_loss.mean()))
                    for ep in range(config.TESTING_NUM):
                        batch_xs_training,batch_ys_training = next(training_data)
                        train_src_loss[ep], train_pred_loss[ep] = self.sess.run(
                            [self.validation_srcing_loss, self.validation_preding_loss],
                            feed_dict={self.validation_images: batch_xs_training,
                                       self.validation_labels: batch_ys_training})
                    print('epoch: %d ,ave_training_src_MSE: %.7f,ave_training_pred_MSE: %.7f' % ( epoch, train_src_loss.mean(), train_pred_loss.mean()))
                    if test_pred_loss.mean() < best_mse:
                        save_path = self.saver.save(self.sess, config.save_model_filename_best)
                        best_mse = test_pred_loss.mean()
                        ear_stop = early_stop_number
                        best_epoch = epoch
                    else:
                        ear_stop -= 1
                    print('best_epoch: %d, ear_stop: %d' % (best_epoch, ear_stop))
                    if ear_stop == 0:
                        print('best_mse: %.7f' % (best_mse))
                        break
                    log = "Best_epoch: {}\n Epoch: {}\n mse val: {:8}\n mse_src: {:8}\n training_val_loss: {:8}\n training_src_loss:{:8}".format(
                    best_epoch,
                    epoch + 1,
                    test_pred_loss.mean(),
                    test_src_loss.mean(),
                    train_pred_loss.mean(),
                    train_src_loss.mean()
                    )
                    print(log)
                    self.log_all.debug(log)

    def pred_test(self, config):
        test_data = self.data_input(config,config.training_filename)
        self.saver.restore(self.sess, config.save_model_filename_best)
        test_src_mse = np.zeros((config.TESTING_NUM, 1))
        test_pred_mse = np.zeros((config.TESTING_NUM, 1))
        recon_6channel = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')
        high_res_images = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')
        low_res_images = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')

        for ep in range(config.TESTING_NUM):
            batch_xs_validation, batch_ys_validation = next(test_data)
            recon_6channel[:,:,ep,:], high_res_images[:,:,ep,:], low_res_images[:,:,ep,:] = self.sess.run([self.validation_pred, self.validation_labels, self.validation_images],
                                                              feed_dict={self.validation_images: batch_xs_validation,
                                                                         self.validation_labels: batch_ys_validation})
            test_src_mse[ep], test_pred_mse[ep] = self.sess.run(
                [self.validation_srcing_loss, self.validation_preding_loss],
                feed_dict={self.validation_images: batch_xs_validation,
                           self.validation_labels: batch_ys_validation})

        print('ave_src_MSE: %.7f,ave_pred_MSE: %.7f' % (test_src_mse.mean(), test_pred_mse.mean()))
        saving_path = 'test_results'
        tl.files.exists_or_mkdir(saving_path)
        sio.savemat(os.path.join(saving_path, 'recon_6channel.mat'), {'recon_6channel': recon_6channel})
        sio.savemat('low_res_images.mat', {'low_res_images': low_res_images})
        sio.savemat('high_res_images.mat', {'high_res_images': high_res_images})
