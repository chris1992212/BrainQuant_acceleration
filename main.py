import tensorflow as tf
import os
from model import BrainQuantAI_Part_one
from easydict import EasyDict as edict
import json

# data path and log path
config = edict()
config.testing = edict()
# model
config.MODEL_NAME = 'Dense'          # srcnn, vgg7, vgg_deconv_7,SRresnet,u_net,EDSR
config.INPUT_SIZE = 384                       # the image size input to the network
config.PE_size_ori = 288
config.FE_size_ori = 384
config.Scale =1
config.LEARNING_RATE = 0.0001
config.epoch = 250000
config.batch_size = 16
config.DISPLAY_STEP = 10
config.image_size_FE = 80
config.image_size_PE = 80

config.label_size = 80
config.test_PE = 288
config.test_FE = 384
config.early_stop_number = 1000
config.c_dim = 6
config.is_train = True

# testing
config.testing.patch_size_PE = 288
config.testing.patch_size_FE = 384
config.testing_BN = False

#### Filename

config.training_filename = os.path.join('H:\BrainQuantAI_Data\\simulated_4_2\\xujun\\norm_for_tensorflow\\6channel')
config.testing_filename = os.path.join('H:\BrainQuantAI_Data\simulated_4_2\hym\\norm_for_tensorflow')
config.Test_filename2 = os.path.join('train','6_channel_hanlu')
config.tfrecord_train = os.path.join('Amp_6channel.tfrecord')
config.restore_model_filename = os.path.join('Good_model_for_Amp','model_Amp_6channel_bn_7_12','u_net_bn_new_2','good','mymodel')
config.save_model_filename = os.path.join('Model_11_26','mymodel.ckpt')

config.save_model_filename_best = os.path.join('Model_11_26','good','mymodel.ckpt')
# testing for all pictures
config.Test_Batch_size = 1
config.TESTING_NUM = int(48/config.Test_Batch_size)
config.log_dir = "log_{}".format(config.MODEL_NAME)

###
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")



def main(_): #?
    with tf.Session() as sess:
        brainquant = BrainQuantAI_Part_one(sess,
                      image_size_FE = config.image_size_FE,
                       image_size_PE=config.image_size_PE,
                       label_size = config.label_size,
                      is_train = config.is_train,
                      batch_size = config.batch_size,
                      c_dim = config.c_dim,
                      test_FE = config.test_FE,
                      test_PE= config.test_PE
                         )

        brainquant.train(config)
        # brainquant.pred_test(config)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ' 1'
    tf.app.run()