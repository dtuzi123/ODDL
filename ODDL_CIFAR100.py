import time
from utils import *
#from scipy.misc import imsave as ims
#from ops import *
#from utils import *
#from Utlis2 import *
import random as random
from glob import glob
import os, gzip
from glob import glob
from Basic_structure import *
from mnist_hand import *
from CIFAR10 import *
#import keras as K
import tensorflow.keras as K

from tensorflow.keras import layers
#from skimage.measure import compare_ssim
import skimage as skimage
from tensorflow_probability import distributions as tfd
from ResNet18_Small import *

#os.environ["CUDA_VISIBLE_DEVICES"] = '6'

import numpy.linalg as la
#import Fid_tf2 as fid2
#from inception import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from mnist_hand import *
import tensorflow as tf2

def file_name(file_dir):
    t1 = []
    file_dir = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)
    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def Classifier(name, image, z_dim=20, reuse=False):
    with tf.compat.v1.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        image = tf.compat.v1.reshape(image, (-1, 28 * 28))
            #image = tf.compat.v1.concat((image, y), axis=1)
        net = tf.compat.v1.nn.relu(bn(linear(image, 400, scope='g_fc1'), is_training=True, scope='g_bn1'))

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(image, 400, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 10
        logoutput = linear(h5, continous_len, 'e_log_sigma_sq')

        return logoutput


def Image_classifier(inputs, scopename, is_training=True, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(inputs, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

def CodeImage_classifier(s, scopename, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        input = s

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        n_hidden = 500
        keep_prob = 0.9

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 4
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def my_gumbel_softmax_sample(logits, cats_range, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    logits_with_noise = tf.nn.softmax(y / temperature)
    return logits_with_noise


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def My_Encoder_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def My_Classifier_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        # z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue


def MINI_Classifier(s, scopename, reuse=False):
    keep_prob = 1.0
    with tf.variable_scope(scopename, reuse=reuse):
        input = s
        n_output = 10
        n_hidden = 500
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 10
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y


# Create model of CNN with slim api

class LifeLone_MNIST(object):
    def __init__(self):
        self.data_stream_batch = 10
        self.batch_size = 64
        self.input_height = 32
        self.input_width = 32
        self.c_dim = 3
        self.z_dim = 200
        self.len_discrete_code = 4
        self.epoch = 10

        self.learning_rate = 0.0001
        self.classifier_learningRate = 0.001

        self.beta1 = 0.5
        self.beta = 1.0

        self.maxMemorySize = 5000

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

        arr1, arr2 = Split_DataSet_CIFAR100(x_train, y_train)
        self.arr1_testing, self.arr2_testing = Split_DataSet_CIFAR100_Testing(x_test, y_test)

        arr1 = np.array(arr1)
        arr2 = np.array(arr2)

        self.arr1 = arr1 / 255.0
        self.arr2 = keras.utils.to_categorical(arr2, 100)

        print(np.shape(self.arr2))
        self.totalSet = self.arr1
        self.totalLabel = self.arr2


        self.superEncoderArr = []
        self.subEncoderArr = []
        self.superGeneratorArr = []
        self.subGeneratorArr = []
        self.zArr = []
        self.latentXArr = []
        self.KLArr = []

        self.recoArr = []
        self.componentCount = 0

        self.lossArr = []
        self.memoryArr = []

        self.parameterArr = []
        self.label = tf.placeholder(tf.float32, [self.batch_size, 10])
        self.KDlabel = tf.placeholder(tf.float32, [self.batch_size, 10])

        self.StudentLogitInput = tf.placeholder(tf.float32, [self.batch_size, 10])

        self.ClassifierPrediction = []
        self.ClassifierLogits = []
        self.Give_Feature = 0
        self.GeneratorArr = []
        self.TeacherFeatures = []
        self.TeacherPredictions = []
        self.TeacherLossArr = []

        self.classifierModelArr = []
        self.TeacherOptimArr = []

    def Random_Data(self,data):
        n_examples = np.shape(data)[0]
        index = [i for i in range(n_examples)]
        random.shuffle(index)
        data = data[index]
        return data

    def Create_Student(self,index,selectedComponentIndex):
        beta = 0.01

        SuperGeneratorStr = "SuperGenerator" + str(index)
        SubGeneratorStr = "SubGenerator" + str(index)

        SuperEncoder = "SuperEncoder" + str(index)
        SubEncoder = "SubEncoder" + str(index)
        classifierName = "StudentClassifier" + str(index)

        is_training = True

        # Classifier
        net = ResNet([2, 2, 2, 2], 100)
        net.compile(tf.keras.optimizers.SGD(self.classifier_learningRate),
                    loss=tf2.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        self.StudentClassifier = net

    def Create_subloss(self, G, name):
        name = "discriminator1"
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G
        d_hat = Discriminator_SVHN_WGAN(x_hat, name, reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        return ddx

    def shoaared_encoder(self,name, x, z_dim=20, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.tanh)
        return l1

    def encoder(self,name, x, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.tanh)
            #l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            mu = tf.layers.dense(l1, z_dim, activation=None)
            sigma = tf.layers.dense(l1, z_dim, activation=tf.exp)
            return mu, sigma

    def shared_decoder(self,name,z, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.tanh)
            return l1

    def decoder(self,name,z, z_dim=50, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.relu)
            #l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            x_hat = tf.layers.dense(
                l1, self.data_dim, activation=None)
            return x_hat

    def Create_VAEs(self,index,selectedComponentIndex):
        beta = 0.01

        SuperGeneratorStr = "SuperGenerator" + str(index)
        SubGeneratorStr = "SubGenerator" + str(index)

        SuperEncoder = "SuperEncoder" + str(index)
        SubEncoder = "SubEncoder" + str(index)
        classifierName = "Classifier" + str(index)
        discriminatorName = "discriminator" + str(index)
        generatorName = "GAN_generator" + str(index)

        is_training = True

        sharedEncoderName = "sharedEncoder" + str(index)
        encoderName = "Encoder" + str(index)
        sharedDecoderName = "sharedDecoder" + str(index)
        decoderName = "Decoder" + str(index)

        if self.componentCount == 0:
            # Classifier
            net = ResNet([2, 2, 2, 2], 100)
            net.compile(tf.keras.optimizers.SGD(self.classifier_learningRate),
                        loss=tf2.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            self.classifierModelArr.append(net)

            # Building the first component
            z_mean_super, z_log_sigma_super = CIFAR_SuperEncoder(self.inputs, SuperEncoder, is_training, batch_size=64,
                                                                 reuse=False)
            # z_mean_super = CIFAR_SuperEncoder(self.inputs, SuperEncoder, is_training, batch_size=64, reuse=False)
            continous_variables1_super = z_mean_super + z_log_sigma_super * tf.random_normal(tf.shape(z_mean_super), 0,
                                                                                             1, dtype=tf.float32)

            z_mean_sub, z_log_sigma_sub = CIFAR_SubEncoder(continous_variables1_super, SubEncoder, is_training,
                                                           batch_size=64, reuse=False)
            continous_variables1_sub = z_mean_sub + z_log_sigma_sub * tf.random_normal(tf.shape(z_mean_sub), 0, 1,
                                                                                       dtype=tf.float32)

            z_mean_generator = CIFAR_SuperGenerator(continous_variables1_sub, SuperGeneratorStr, is_training,
                                                    batch_size=64, reuse=False)

            reco, feature = CIFAR_SubGenerator(z_mean_generator, SubGeneratorStr, is_training, batch_size=64,
                                               reuse=False)
            #generation
            z_mean_generator2 = CIFAR_SuperGenerator(self.z, SuperGeneratorStr, is_training,
                                                    batch_size=64, reuse=True)

            reco2, feature2 = CIFAR_SubGenerator(z_mean_generator2, SubGeneratorStr, is_training, batch_size=64,
                                               reuse=True)
            reco2 = tf.reshape(reco2,(-1,32,32,3))
            self.GeneratorArr.append(reco2)

            self.Give_Feature = feature
            reco = tf.reshape(reco, (-1, 32, 32, 3))
            self.recoArr.append(reco)

            KL_divergence1 = 0.5 * tf.reduce_sum(
                tf.square(z_mean_sub) + tf.square(z_log_sigma_sub) - tf.log(1e-8 + tf.square(z_log_sigma_sub)) - 1, 1)
            KL_divergence1 = tf.reduce_mean(KL_divergence1)

            self.KLArr.append(KL_divergence1)

            reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco - self.inputs), [1, 2, 3]))
            vaeloss1 = reconstruction_loss1 + self.beta * KL_divergence1
            self.recoLoss = reconstruction_loss1
            self.vaeLoss = vaeloss1

            self.lossArr.append(self.vaeLoss)

            T_vars = tf.trainable_variables()

            var1 = [var for var in T_vars if var.name.startswith(SuperEncoder)]
            var2 = [var for var in T_vars if var.name.startswith(SubEncoder)]
            var3 = [var for var in T_vars if var.name.startswith(SuperGeneratorStr)]
            var4 = [var for var in T_vars if var.name.startswith(SubGeneratorStr)]

            VAE_parameters = var1 + var2 + var3 + var4


            with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
                self.VAE_optim1 = tf.train.AdamOptimizer(learning_rate=1e-4) \
                    .minimize(self.vaeLoss, var_list=VAE_parameters)

                self.TeacherOptimArr.append(self.VAE_optim1)

            self.componentCount = self.componentCount + 1
        else:
            # Classifier
            net = ResNet([2, 2, 2, 2], 100)
            net.compile(tf.keras.optimizers.SGD(self.learning_rate),
                        loss=tf2.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            self.classifierModelArr.append(net)

            # Building the first component
            z_mean_super, z_log_sigma_super = CIFAR_SuperEncoder(self.inputs, SuperEncoder, is_training, batch_size=64,
                                                                 reuse=False)
            # z_mean_super = CIFAR_SuperEncoder(self.inputs, SuperEncoder, is_training, batch_size=64, reuse=False)
            continous_variables1_super = z_mean_super + z_log_sigma_super * tf.random_normal(tf.shape(z_mean_super), 0,
                                                                                             1, dtype=tf.float32)

            z_mean_sub, z_log_sigma_sub = CIFAR_SubEncoder(continous_variables1_super, SubEncoder, is_training,
                                                           batch_size=64, reuse=False)
            continous_variables1_sub = z_mean_sub + z_log_sigma_sub * tf.random_normal(tf.shape(z_mean_sub), 0, 1,
                                                                                       dtype=tf.float32)

            z_mean_generator = CIFAR_SuperGenerator(continous_variables1_sub, SuperGeneratorStr, is_training,
                                                    batch_size=64, reuse=False)

            reco, feature = CIFAR_SubGenerator(z_mean_generator, SubGeneratorStr, is_training, batch_size=64,
                                               reuse=False)

            #generation
            z_mean_generator2 = CIFAR_SuperGenerator(self.z, SuperGeneratorStr, is_training,
                                                    batch_size=64, reuse=True)

            reco2, feature2 = CIFAR_SubGenerator(z_mean_generator2, SubGeneratorStr, is_training, batch_size=64,
                                               reuse=True)
            reco2 = tf.reshape(reco2,(-1,32,32,3))
            self.GeneratorArr.append(reco2)

            self.Give_Feature = feature
            reco = tf.reshape(reco, (-1, 32, 32, 3))
            self.recoArr.append(reco)

            KL_divergence1 = 0.5 * tf.reduce_sum(
                tf.square(z_mean_sub) + tf.square(z_log_sigma_sub) - tf.log(1e-8 + tf.square(z_log_sigma_sub)) - 1, 1)
            KL_divergence1 = tf.reduce_mean(KL_divergence1)

            self.KLArr.append(KL_divergence1)

            reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco - self.inputs), [1, 2, 3]))
            vaeloss1 = reconstruction_loss1 + self.beta * KL_divergence1
            self.recoLoss = reconstruction_loss1
            self.vaeLoss = vaeloss1

            self.lossArr.append(self.vaeLoss)

            T_vars = tf.trainable_variables()
            var1 = [var for var in T_vars if var.name.startswith(SuperEncoder)]
            var2 = [var for var in T_vars if var.name.startswith(SubEncoder)]
            var3 = [var for var in T_vars if var.name.startswith(SuperGeneratorStr)]
            var4 = [var for var in T_vars if var.name.startswith(SubGeneratorStr)]

            VAE_parameters = var1 + var2 + var3 + var4

            with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
                self.VAE_optim1 = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
                    .minimize(self.vaeLoss, var_list=VAE_parameters)
                self.TeacherOptimArr.append(self.VAE_optim1)

            self.componentCount = self.componentCount + 1

            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            self.sess.run(tf.variables_initializer(not_initialized_vars))

            #set weights for the new component
            '''
            print(selectedComponentIndex)
            parent_SubGeneratorStr = "SubGenerator"+ str(selectedComponentIndex)
            parent_SubEncoder = "SubEncoder"+ str(selectedComponentIndex)
            parent_vars1 = [var for var in T_vars if var.name.startswith(parent_SubGeneratorStr)]
            parent_vars2 = [var for var in T_vars if var.name.startswith(parent_SubEncoder)]

            arr1 = []
            arr1_value = []
            for var in vars1:
                arr1.append(var)

            for var in parent_vars1:
                arr1_value.append(var)

            print(np.shape(arr1))
            print(np.shape(arr1_value))
            for i in range(np.shape(arr1)[0]):
                tf.assign(arr1[i],arr1_value[i])

            arr1 = []
            arr1_value = []

            for var in vars2:
                arr1.append(var)

            for var in parent_vars2:
                arr1_value.append(var)

            for i in range(np.shape(arr1)[0]):
                tf.assign(arr1[i],arr1_value[i])
            '''

    def SelectComponentByData(self,data):
        mycount = int(np.shape(data)[0] / self.batch_size)
        losses = []
        for i in range(np.shape(self.lossArr)[0]):
            sumLoss = 0
            for j in range(mycount):
                batch = data[j * self.batch_size:(j + 1) * self.batch_size]
                loss1 = self.sess.run(self.lossArr[i],feed_dict={self.inputs:batch})
                sumLoss = sumLoss + loss1
            sumLoss = sumLoss / mycount
            losses.append(sumLoss)

        #print("index")
        #print(losses)
        losses = np.array(losses)
        index = np.argmin(losses)
        #index = index + 1

        return index

    def Reconstruction_ByIndex(self,index,test):

        count = int(np.shape(test)[0] / self.batch_size)
        realTest = []
        recoArr = []
        for i in range(count):
            batch = test[i * self.batch_size : (i+1) * self.batch_size]
            reco = self.sess.run(self.recoArr[index],feed_dict={self.inputs:batch})
            for j in range(self.batch_size):
                realTest.append(batch[j])
                recoArr.append(reco[j])

        realTest = np.array(realTest)
        recoArr  = np.array(recoArr)

        return realTest,recoArr

    def ClassficationEvaluation(self):
        index1 = self.SelectComponentByData(self.train_arr1)
        test1 = self.train_arr1
        test1, reco1 = self.Reconstruction_ByIndex(index1, test1)
        label1 = self.train_labelArr1[0:np.shape(reco1)[0]]

        index2 = self.SelectComponentByData(self.train_arr2)
        test2 = self.train_arr2
        test2, reco2 = self.Reconstruction_ByIndex(index2, test2)
        label2 = self.train_labelArr2[0:np.shape(reco2)[0]]

        index3 = self.SelectComponentByData(self.train_arr3)
        test3 = self.train_arr3
        test3, reco3 = self.Reconstruction_ByIndex(index3, test3)
        label3 = self.train_labelArr3[0:np.shape(reco3)[0]]

        index4 = self.SelectComponentByData(self.train_arr4)
        test4 = self.train_arr4
        test4, reco4 = self.Reconstruction_ByIndex(index4, test4)
        label4 = self.train_labelArr4[0:np.shape(reco4)[0]]

        index5 = self.SelectComponentByData(self.train_arr5)
        test5 = self.train_arr5
        test5, reco5 = self.Reconstruction_ByIndex(index5, test5)
        label5 = self.train_labelArr5[0:np.shape(reco5)[0]]

        index6 = self.SelectComponentByData(self.train_arr6)
        test6 = self.train_arr6
        test6, reco6 = self.Reconstruction_ByIndex(index6, test6)
        label6 = self.train_labelArr6[0:np.shape(reco6)[0]]

        index7 = self.SelectComponentByData(self.train_arr7)
        test7 = self.train_arr7
        test7, reco7 = self.Reconstruction_ByIndex(index7, test7)
        label7 = self.train_labelArr7[0:np.shape(reco7)[0]]

        index8 = self.SelectComponentByData(self.train_arr8)
        test8 = self.train_arr8
        test8, reco8 = self.Reconstruction_ByIndex(index8, test8)
        label8 = self.train_labelArr8[0:np.shape(reco8)[0]]

        index9 = self.SelectComponentByData(self.train_arr9)
        test9 = self.train_arr9
        test9, reco9 = self.Reconstruction_ByIndex(index9, test9)
        label9 = self.train_labelArr8[0:np.shape(reco9)[0]]

        index10 = self.SelectComponentByData(self.train_arr10)
        test10 = self.train_arr10
        test10, reco10 = self.Reconstruction_ByIndex(index10, test10)
        label10 = self.train_labelArr10[0:np.shape(reco10)[0]]

        arrLabels = []
        arrReco = []

        arrLabels.append(label1)
        arrLabels.append(label2)
        arrLabels.append(label3)
        arrLabels.append(label4)
        arrLabels.append(label5)
        arrLabels.append(label6)
        arrLabels.append(label7)
        arrLabels.append(label8)
        arrLabels.append(label9)
        arrLabels.append(label10)

        arrReco.append(reco1)
        arrReco.append(reco2)
        arrReco.append(reco3)
        arrReco.append(reco4)
        arrReco.append(reco5)
        arrReco.append(reco6)
        arrReco.append(reco7)
        arrReco.append(reco8)
        arrReco.append(reco9)
        arrReco.append(reco10)

        arrLabels, arrReco = self.Combined_data(arrLabels, arrReco)

        from keras.utils import np_utils
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras.optimizers import RMSprop

        model = Sequential([
            Dense(2048, input_dim=32*32*3),  # input_dim即为28* 28=784，output_dim为32，即传出来只有32个feature
            Activation('relu'),  # 变成非线性化的数据
            Dense(1024, input_dim=2048),
            Activation('relu'),  # 变成非线性化的数据
            Dense(512, input_dim=1024),
            Activation('relu'),  # 变成非线性化的数据
            Dense(256, input_dim=512),
            Activation('relu'),  # 变成非线性化的数据
            Dense(10),  # input即为上一层的output，故定义output_dim是10个feature就可以
            Activation('softmax')  # 使用softmax进行分类处理
        ])

        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(  # 激活model
            optimizer=rmsprop,  # 若是要使用默认的参数，即optimizer=‘rmsprop'即可
            loss='categorical_crossentropy',  # crossentropy协方差
            metrics=['accuracy'])

        arrReco = np.reshape(arrReco,(-1,32*32*3))
        test = np.reshape(self.cifar_test_x,(-1,32*32*3))
        model.fit(arrReco, arrLabels, epochs=100, batch_size=64)  # 使用fit功能来training；epochs表示训练的轮数；
        loss, accuracy = model.evaluate(test, self.cifar_test_label)
        return accuracy

    def build_model(self):
        min_value = 1e-10
        # some parameters
        bs = self.batch_size

        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims)
        self.KDinputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims)

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])
        self.weights = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.index = tf.placeholder(tf.int32, [self.batch_size])
        self.gan_inputs = tf.placeholder(tf.float32, [1] + image_dims)
        self.gan_domain = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.gan_domain_labels = tf.placeholder(tf.float32, [self.batch_size, 1])

        # GAN networks
        self.Create_VAEs(1,0)
        #self.Create_Student(1,0)

    def Generate_PreviousSamples(self, num):
        b_num = int(num / self.batch_size)
        mylist = []
        for i in range(b_num):
            # update GAN
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            gan1 = self.sess.run(
                self.GAN_gen1,
                feed_dict={self.z: batch_z})
            print(np.shape(gan1))
            for ttIndex in range(self.batch_size):
                mylist.append(gan1[ttIndex])
        mylist = np.array(mylist)
        return mylist

    def Combined_data(self,arr1,arr2):
        r1 = []
        r2 = []
        for i in range(np.shape(arr1)[0]):
            b1 = arr1[i]
            b2 = arr2[i]
            for j in range(np.shape(b1)[0]):
                r1.append(b1[j])
                r2.append(b2[j])

        r1 = np.array(r1)
        r2 = np.array(r2)
        return r1,r2

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "label_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def Give_predictedLabels(self, testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            predictions = np.argmax(predictions,axis=1)
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = K.utils.to_categorical(totalPredictions)
        return totalPredictions

    def domain_predict(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)
        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)

        domain_logit = tf.nn.softmax(domain_logit)
        predictions = tf.argmax(domain_logit, 1)
        return predictions

    def Give_RealReconstruction(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        return reconstruction_loss1

    def Calculate_ReconstructionErrors(self, testX):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Give_RealReconstruction()
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError


    def random_pick(self, some_list, probabilities):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability: break
        return item

    def Regular_Matrix(self,matrix):
        count = np.shape(matrix)[0]
        #avoid negative value for each element
        for i in range(count):
            for j in range(count):
                if matrix[i,j] < 0:
                    matrix[i,j] = 0.05

        #normalize the probability for each row
        for i in range(count):
            row = matrix[i,0:count]
            sum1 = self.ReturnSum(row)
            for j in range(count):
                row[j] = row[j] / sum1
                self.proMatrix[i,j] = row[j]


    def ReturnSum(self,arr):
        count = np.shape(arr)[0]
        sum1 = 0
        for i in range(count):
            sum1 += arr[i]

        return sum1

    def SelectComponentByBatch(self,batch):
        losses = []
        for j in range(self.batch_size-1):
            batch = np.concatenate((batch,batch),axis=0)
        batch = np.array(batch)

        for i in range(np.shape(self.lossArr)[0]):
            loss1 = self.sess.run(self.lossArr[i], feed_dict={self.inputs: batch})
            losses.append(loss1)

        print("index")
        print(losses)
        losses = np.array(losses)
        index = np.argmin(losses)
        #index = index + 1

        return index,batch

    def Give_Reconstruction_ByAnySamples(self,set):
        recoList = []
        count = np.shape(set)[0]
        for i in range(count):
            batch = set[i]
            index2,batch2 = self.SelectComponentByBatch(batch)
            reco = self.sess.run(self.recoArr[index2 - 1],feed_dict={self.inputs:batch2})
            recoList.append(reco[0])

        recoList = np.array(recoList)
        return recoList

    def KeepSize(self,arr,size):
        mycount = np.shape(arr)[0]
        mycount = int(mycount/self.batch_size)
        lossArr = []
        arr = np.array(arr)
        for i in range(mycount):
            batch = arr[i*self.batch_size : (i+1)*self.batch_size]
            loss = self.sess.run(self.VAE_multi,feed_dict={self.inputs:batch})
            for j in range(self.batch_size):
                lossArr.append(loss[j])

        lossArr = np.array(lossArr)
        index = np.argsort(lossArr)
        print(index)
        index2 = [int(i) for i in index]
        arr = arr[index2]
        if np.shape(arr)[0] < size:
            size = np.shape(arr)[0]
        arr = arr[0:size]

        arr2 = []
        for i in range(size):
            arr2.append(arr[i])

        return arr2

    def FID_Evaluation(self,recoArr,test):

        recoArr = np.array(recoArr)
        test = np.array(test)
        fid2.session = self.sess

        test1 = np.transpose(test, (0, 3, 1, 2))
        # test1 = ((realImages + 1.0) * 255) / 2.0
        test1 = test1 * 255.0

        test2 = np.transpose(recoArr, (0, 3, 1, 2))
        # test1 = ((realImages + 1.0) * 255) / 2.0
        test2 = test2 * 255.0

        fidScore = fid2.get_fid(test1, test2)
        print(fidScore)

    def Calculate_FID_Score(self,test1,test2):
        fid2.session = self.sess

        test1 = np.transpose(test1, (0, 3, 1, 2))
        #test1 = ((realImages + 1.0) * 255) / 2.0
        test1 = test1 * 255.0

        test2 = np.transpose(test2, (0, 3, 1, 2))
        #test1 = ((realImages + 1.0) * 255) / 2.0
        test2 = test2 * 255.0

        count = int(np.shape(test1)[0] / self.batch_size)
        fidSum = 0
        for i in range(count):
            realX = test1[i*self.batch_size:(i+1)*self.batch_size]
            fidScore = fid2.get_fid(realX, test2)
            fidSum = fidSum + fidScore
        fidSum = fidSum / count

        return fidSum

    def Evaluation(self,recoArr,test):
        count = int(np.shape(test)[0] / self.batch_size)
        ssimSum = 0
        psnrSum = 0
        mseSum = 9
        for i in range(count):
            batch = test[i * self.batch_size:(i + 1) * self.batch_size]
            reco = recoArr[i * self.batch_size:(i + 1) * self.batch_size]
            mySum2 = 0
            # Calculate SSIM
            for t in range(self.batch_size):
                # ssim_none = ssim(g[t], r[t], data_range=np.max(g[t]) - np.min(g[t]))
                ssim_none = compare_ssim(batch[t], reco[t], multichannel=True)
                mySum2 = mySum2 + ssim_none
            mySum2 = mySum2 / self.batch_size
            ssimSum = ssimSum + mySum2

            # Calculate PSNR
            mySum2 = 0
            for t in range(self.batch_size):
                measures = skimage.measure.compare_psnr(batch[t], reco[t], data_range=np.max(reco[t]) - np.min(reco[t]))
                mySum2 = mySum2 + measures
            mySum2 = mySum2 / self.batch_size
            psnrSum = psnrSum + mySum2

            # Calculate MSE
            mySum2 = 0
            for t in range(self.batch_size):
                # ssim_none = ssim(g[t], r[t], data_range=np.max(g[t]) - np.min(g[t]))
                mse = skimage.measure.compare_mse(batch[t], reco[t])
                mySum2 = mySum2 + mse
            mySum2 = mySum2 / self.batch_size
            mseSum = mseSum + mySum2

        ssimSum = ssimSum / count
        psnrSum = psnrSum / count
        mseSum = mseSum / count

        # Calculate InscoreScore
        count = np.shape(test)[0]
        count = int(count / self.batch_size)
        realArray = []
        array = []
        for i in range(count):
            x_fixed = test[i * self.batch_size:(i + 1) * self.batch_size]
            yy = recoArr[i * self.batch_size:(i + 1) * self.batch_size]

            yy = yy * 255
            yy = np.reshape(yy, (-1, 32, 32, 3))

            for t in range(self.batch_size):
                array.append(yy[t])
                realArray.append(x_fixed[t])

        real1 = realArray
        score = get_inception_score(array)

        return ssimSum,psnrSum,score

    def Give_Features_Function(self,test):
        count = np.shape(test)[0]
        newCount = int(count / self.batch_size)
        remainCount = count - newCount * self.batch_size
        remainSamples = test[newCount * self.batch_size:count]
        remainSamples = np.concatenate((remainSamples, test[0:(self.batch_size - remainCount)]), axis=0)
        totalSamples = test

        featureArr = []
        for i in range(newCount):
            batch = totalSamples[i * self.batch_size:(i + 1) * self.batch_size]
            features = self.sess.run(self.Give_Feature, feed_dict={self.inputs: batch})
            for j in range(self.batch_size):
                featureArr.append(features[j])

        ff = self.sess.run(self.Give_Feature, feed_dict={self.inputs: remainSamples})
        for i in range(remainCount):
            featureArr.append(ff[i])

        featureArr = np.array(featureArr)
        return featureArr

    def Predictions_By_Index2(self,testX,index):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)

        #Selection
        lossArr = []
        for t2 in range(np.shape(self.lossArr)[0]):
            sumLoss = 0
            for t1 in range(myN):
                newbatch = testX[t1*self.batch_size:(t1+1)*self.batch_size]
                loss1 = self.sess.run(self.lossArr[t2],feed_dict={self.inputs:newbatch})
                sumLoss = sumLoss + loss1
            sumLoss = sumLoss / myN
            lossArr.append(sumLoss)

        minIndex = np.argmin(lossArr)

        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            predictions = self.classifierModelArr[minIndex].predict(my1)
            predictions = np.argmax(predictions,axis=1)
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])
        totalPredictions = np.array(totalPredictions)
        return totalPredictions

    def SelectComponentBySingle(self,single):
        x_test = np.tile(single,(self.batch_size,1,1,1))
        lossArr = []
        for i in range(self.componentCount):
            loss = self.sess.run(self.lossArr[i],feed_dict={self.inputs:x_test})
            lossArr.append(loss)
        index = np.argmin(lossArr)
        return x_test,index

    def Calculate_Accuracy_ByALL(self, testX,testY, index):
        totalCount = np.shape(testX)[0]
        myPro = []
        for i in range(totalCount):
            single = testX[i]
            single = np.reshape(single,(1,32,32,3))

            x,index = self.SelectComponentBySingle(single)
            pred = self.classifierModelArr[index].predict(x)
            pred = np.argmax(pred,axis=1)
            pred = pred[0]
            myPro.append(pred)

        target = [np.argmax(one_hot) for one_hot in testY]
        sumError = 0
        accCount = 0
        for i in range(np.shape(myPro)[0]):
            isState = True

            if myPro[i] == target[i]:
                accCount = accCount + 1

        totalCount = np.shape(myPro)[0]
        acc = float(accCount / totalCount)

        return acc

    def Calculate_Discrepancy_Single(self,index,single):

        single2 = np.reshape(single,(1,32,32,3))
        batch = np.tile(single2,(self.batch_size,1,1,1))
        pred1 = self.classifierModelArr[index].predict(batch)
        pred2 = self.classifierModelArr[self.componentCount-1].predict(batch)
        pred1 = pred1[0]
        pred2 = pred2[0]
        pred1 = np.argmax(pred1)
        pred2 = np.argmax(pred2)

        diff = np.abs(pred1 - pred2)
        return diff

    def Calculate_Accuracy_ByIndex(self, testX,testY, index):

        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Predictions_By_Index2(testX,index)
        target = [np.argmax(one_hot)for one_hot in testY]
        sumError = 0
        accCount = 0
        for i in range(np.shape(myPro)[0]):
            isState = True

            if myPro[i] == target[i]:
                accCount = accCount + 1

        totalCount = np.shape(myPro)[0]
        acc = float(accCount/totalCount)

        return acc

    def gaussian(self,sigma,x,y):
        return np.exp(-np.sqrt(la.norm(x - y) ** 2 / (2 * sigma ** 2)))

    def Give_Features_Function2(self,test):
        count = np.shape(test)[0]
        totalSamples = test

        featureArr = []
        for i in range(count):
            single = totalSamples[i]
            single = np.reshape(single,(1,32,32,3))
            feature = self.sess.run(self.Give_Feature2, feed_dict={self.input_test: single})
            featureArr.append(feature)

        featureArr = np.array(featureArr)
        return featureArr

    def SelectSample_InMemory(self):
        sigma = 10
        dynamicFeatureArr = self.Give_Features_Function2(self.DynamicMmeory)
        fixedFeatureArr = self.Give_Features_Function2(self.FixedMemory)

        count = np.shape(dynamicFeatureArr)[0]
        count2 = np.shape(fixedFeatureArr)[0]
        relationshipMatrix = np.zeros((count,count2))
        for i in range(count):
            for j in range(count2):
                relationshipMatrix[i,j] = self.gaussian(sigma,dynamicFeatureArr[i],fixedFeatureArr[j])

        sampleDistance = []
        for i in range(count):
            sum1 = 0
            for j in range(count2):
                sum1 = sum1 + relationshipMatrix[i, j]
            sum1 = sum1 / count2
            sampleDistance.append(sum1)

        sampleDistance = np.array(sampleDistance)
        index = np.argsort(-sampleDistance)
        self.DynamicMmeory = self.DynamicMmeory[index]
        self.DynamicMmeoryLabel = self.DynamicMmeoryLabel[index]
        sampleDistance = sampleDistance[index]

        print(sampleDistance)

        print(self.ThresholdForFixed)
        if np.shape(self.FixedMemory)[0] < self.maxMmeorySize * 1000:
            print("diff")
            for i in range(count):
                if i > 15:
                    break

                if sampleDistance[i] > self.ThresholdForFixed:
                    single = self.DynamicMmeory[i]
                    single = np.reshape(single,(1,32,32,3))

                    singleLabel = self.DynamicMmeoryLabel[i]
                    singleLabel = np.reshape(singleLabel,(1,-1))

                    self.FixedMemory = np.concatenate((self.FixedMemory,single),axis=0)
                    self.FixedMemoryLabel = np.concatenate((self.FixedMemoryLabel,singleLabel),axis=0)

                    print(sampleDistance[i])
                else:
                    break

    def Calculate_Acc(self,test,testLabel):

        #target = [np.argmax(one_hot)for one_hot in testLabel]
        target = testLabel
        count = np.shape(test)[0]
        realTest = []
        recoArr = []
        accCount = 0
        totalCount = 0
        index = self.SelectComponentByData(test)

        count = int(count/self.batch_size)
        for i in range(count):
            batch = test[i * self.batch_size:(i+1)*self.batch_size]
            baatchLabel = target[i * self.batch_size:(i+1)*self.batch_size]

            prediction = self.classifierModelArr[index].predict(batch)
            prediction = np.argmax(prediction,axis=1)
            #prediction = self.sess.run(self.predictionArr[index], feed_dict={self.input_test: single})
            for tt in range(self.batch_size):
                if prediction[tt] == baatchLabel[tt]:
                    accCount = accCount + 1
                totalCount = totalCount + 1

        acc = float(accCount/totalCount)
        return acc


    def Calculate_Acc2(self):
        totalAcc = 0
        for i in range(20):
            test = self.arr1_testing[i]
            testy = self.arr2_testing[i]
            test = np.array(test)
            testy = np.array(testy)

            acc = self.Calculate_Acc(test,testy)
            totalAcc = totalAcc + acc
        totalAcc = totalAcc / 20
        return totalAcc


    def train(self):

        taskCount = 1

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config.gpu_options.allow_growth = True

        self.MemoryArr = []

        isFirstStage = True
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.Create_VAEs(self.componentCount+1,0)
            # self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion_invariant')

            # saver to save model
            self.saver = tf.train.Saver()
            ExpertWeights = np.ones((self.batch_size, 4))
            DomainState = np.zeros(4).astype(np.int32)
            DomainState[0] = 0
            DomainState[1] = 1
            DomainState[2] = 2
            DomainState[3] = 3

            MnistRecoArray = []
            FashionRecoArray = []
            SvhnRecoArray = []
            IFashionRecoArray = []
            AverageRecoArray = []
            Single_AverageSourceRisk = []

            self.totalMemory = []
            self.totalMemoryLabel = []

            self.totalSet = np.array(self.totalSet)
            self.totalLabel = np.array(self.totalLabel)

            #self.totalSet = np.reshape(self.totalSet, (-1, 28 * 28))

            self.FixedMemory = self.totalSet[0:self.batch_size]
            self.FixedMemoryLabel = self.totalLabel[0:self.batch_size]
            self.FixedMemory = np.array(self.FixedMemory)
            self.FixedMemoryLabel = np.array(self.FixedMemoryLabel)

            self.totalMemory.append(self.FixedMemory)
            self.totalMemoryLabel.append(self.FixedMemoryLabel)

            self.ThresholdForFixed = 1

            self.minThreshold = 0.0005
            self.maxThreshold = 0.05

            self.DynamicMmeory = self.totalSet[0:self.batch_size]
            self.DynamicMmeoryLabel = self.totalLabel[0:self.batch_size]

            self.maxMmeorySize = 5000

            self.DynamicMmeory = np.array(self.DynamicMmeory)
            self.DynamicMmeoryLabel = np.array(self.DynamicMmeoryLabel)

            totalCount = int(np.shape(self.totalSet)[0] / self.data_stream_batch)

            self.moveThreshold = (self.maxThreshold - self.minThreshold) / totalCount

            frozenCount = 0

            self.trainingSet = []
            self.trainLabelSet = []

            self.oldX = []
            self.OldY = []

            self.IsKD = 1

            #self.trainingSet.append(self.FixedMemory)
            #self.trainLabelSet.append(self.FixedMemoryLabel)

            '''
            for epoch in range(20):
                n_examples = np.shape(self.totalSet)[0]
                index2 = [i for i in range(n_examples)]
                np.random.shuffle(index2)
                self.totalSet = self.totalSet[index2]
                self.totalSetLabel = self.totalSetLabel[index2]

                for tt in range(int(np.shape(self.totalSet)[0] / self.batch_size)):
                    aaa = self.totalSet[tt*self.batch_size:(tt+1)*self.batch_size]
                    bbb = self.totalSetLabel[tt*self.batch_size:(tt+1)*self.batch_size]
                    _ = self.sess.run(self.Student_Classifier_optim,
                                              feed_dict={self.inputs:aaa,self.label:bbb })

            acc = self.Calculate_Accuracy_ByIndex(self.testX,self.testY,0)
            print("aaa")
            print(acc)
            '''
            trainingStep = 0
            scoreArr = []
            # self.totalSet = self.totalSet[6*self.batch_size:np.shape(self.totalSet)[0]]
            for t1 in range(int(np.shape(self.totalSet)[0] / self.data_stream_batch)):
                newX = self.totalSet[t1 * self.data_stream_batch:(t1 + 1) * self.data_stream_batch]
                newXLabel = self.totalLabel[t1 * self.data_stream_batch:(t1 + 1) * self.data_stream_batch]

                trainingStep = trainingStep + 1

                if np.shape(self.FixedMemory)[0] == 0:
                    self.FixedMemory = newX
                    self.FixedMemoryLabel = newXLabel
                else:
                    self.FixedMemory = np.concatenate((self.FixedMemory, newX), axis=0)
                    self.FixedMemoryLabel = np.concatenate((self.FixedMemoryLabel,newXLabel),axis=0)
                    #self.trainingSet[np.shape(self.trainingSet)[0] -1] = self.FixedMemory
                    #self.trainLabelSet[np.shape(self.trainingSet)[0] -1] = self.FixedMemoryLabel

                if t1 % 5 != 0:
                    continue

                if self.componentCount == 1:

                    epochs = self.epoch
                    # self.Create_Component(2)
                    for epoch in range(epochs):
                        Xtrain_binarized = self.FixedMemory
                        labelArr = self.FixedMemoryLabel

                        n_examples = np.shape(Xtrain_binarized)[0]
                        index2 = [i for i in range(n_examples)]
                        np.random.shuffle(index2)
                        Xtrain_binarized = Xtrain_binarized[index2]
                        labelArr = labelArr[index2]
                        counter = 0

                        myCount = int(np.shape(Xtrain_binarized)[0] / self.batch_size)

                        for idx in range(myCount):
                            batchImages = Xtrain_binarized[idx * self.batch_size:(idx + 1) * self.batch_size]
                            batchLabels = labelArr[idx * self.batch_size:(idx + 1) * self.batch_size]

                            # update GAN
                            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                            _,_, d_loss = self.sess.run([self.TeacherOptimArr[self.componentCount-2],self.TeacherOptimArr[self.componentCount-1], self.vaeLoss],
                                                      feed_dict={self.inputs: batchImages,
                                                                 self.z: batch_z})

                            '''
                            c_loss,_ = self.sess.run([self.StudentClassLoss,self.Student_Classifier_optim],
                            feed_dict = {self.inputs: batchImages, self.label:batchLabels})
                            '''
                    self.classifierModelArr[self.componentCount-1].fit(self.FixedMemory, self.FixedMemoryLabel,
                                                                                          verbose=0, epochs=epochs,
                                                                                          validation_freq=1,
                                                                                          shuffle=True)
                    self.classifierModelArr[self.componentCount-2].fit(self.FixedMemory, self.FixedMemoryLabel,
                                                                                          verbose=0, epochs=epochs,
                                                                                          validation_freq=1,
                                                                                          shuffle=True)
                else:
                    #has more components
                    # self.Create_Component(2)

                    epochs = self.epoch
                    # self.Create_Component(2)
                    for epoch in range(epochs):
                        Xtrain_binarized = self.FixedMemory
                        labelArr = self.FixedMemoryLabel
                        msize = int(np.shape(self.FixedMemory)[0] / self.batch_size)
                        # print(msize)
                        # print(np.shape(self.FixedMemory))

                        n_examples = np.shape(Xtrain_binarized)[0]
                        index2 = [i for i in range(n_examples)]
                        np.random.shuffle(index2)
                        Xtrain_binarized = Xtrain_binarized[index2]
                        labelArr = labelArr[index2]
                        counter = 0

                        myCount = int(np.shape(Xtrain_binarized)[0] / self.batch_size)

                        lossSum = 0
                        for idx in range(myCount):
                            batchImages = Xtrain_binarized[idx * self.batch_size:(idx + 1) * self.batch_size]
                            batchLabels = labelArr[idx * self.batch_size:(idx + 1) * self.batch_size]

                            _, d_loss = self.sess.run([self.TeacherOptimArr[self.componentCount-1], self.vaeLoss],
                                                      feed_dict={self.inputs: batchImages})

                    self.classifierModelArr[self.componentCount-1].fit(self.FixedMemory, self.FixedMemoryLabel,
                                                                                          verbose=0, epochs=epochs,
                                                                                          validation_freq=1,
                                                                                          shuffle=True)

                if np.shape(self.FixedMemory)[0] > self.maxMmeorySize:

                    if self.componentCount == 1:
                        self.Create_VAEs(self.componentCount+1,0)
                        self.FixedMemory = []
                        self.FixedMemoryLabel = []
                    else:
                        #Check Expansion
                        print("Check")
                        disArr = []
                        for kk in range(self.componentCount - 1):
                            diffSum = 0
                            mycount2 = int(np.shape(self.FixedMemory)[0] / self.batch_size)
                            for i in range(mycount2):
                                newBatch = self.FixedMemory[i*self.batch_size:(i+1)*self.batch_size]
                                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                                generatedImages = self.sess.run(self.GeneratorArr[kk],feed_dict={self.z:batch_z})

                                pred1 = self.classifierModelArr[kk].predict(newBatch)
                                pred1 = np.argmax(pred1, axis=1)
                                pred2 = self.classifierModelArr[self.componentCount-1].predict(newBatch)
                                pred2 = np.argmax(pred2, axis=1)

                                pred1_2 = self.classifierModelArr[kk].predict(generatedImages)
                                pred1_2 = np.argmax(pred1_2, axis=1)
                                pred2_2 = self.classifierModelArr[self.componentCount-1].predict(generatedImages)
                                pred2_2 = np.argmax(pred2_2, axis=1)

                                diff1 = np.mean(np.abs(pred1 - pred2))
                                diff2 = np.mean(np.abs(pred1_2 - pred2_2))
                                diff = np.abs(diff1 - diff2)
                                diffSum = diffSum + diff

                            diffSum = diffSum / mycount2
                            disArr.append(diffSum)

                        disMin = np.min(disArr)
                        threshold = 2.0
                        print(disMin)
                        scoreArr.append(disMin)

                        #if trainingStep > 100:
                        if threshold < disMin:
                            #if True:
                            print("Expansion")
                            trainingStep = 0
                            self.Create_VAEs(self.componentCount + 1, 0)
                            self.FixedMemory = []
                            self.FixedMemoryLabel = []
                        else:
                            # perform the sample selection
                            print("Selection")
                            compareMatrix = []
                            totalCount2 = np.shape(self.FixedMemory)[0]
                            for i in range(totalCount2):
                                single = self.FixedMemory[i]
                                # single2 = np.reshape(single, (1, 32,32,3))
                                sum = 0
                                for j in range(self.componentCount - 1):
                                    dis = self.Calculate_Discrepancy_Single(j, single)
                                    sum = sum + dis
                                sum = sum / (self.componentCount - 1)
                                compareMatrix.append(-sum)

                            selectedIndex = np.argsort(compareMatrix)
                            print(np.shape(self.FixedMemory))
                            self.FixedMemory = self.FixedMemory[selectedIndex]
                            self.FixedMemoryLabel = self.FixedMemoryLabel[selectedIndex]
                            self.FixedMemory = self.FixedMemory[0:self.maxMmeorySize]
                            self.FixedMemoryLabel = self.FixedMemoryLabel[0:self.maxMmeorySize]
                            print(np.shape(self.FixedMemory))

                print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                      .format(t1, 0, 0, 0, 1, self.componentCount, 1))
            #
            print("Memory size")
            print(np.shape(self.FixedMemory)[0])
            print("acc")

            print("N of components")
            print(self.componentCount)

            print("my acc")
            acc1 = self.Calculate_Acc2()
            print(acc1)

            socreArr = np.array(scoreArr)
            arr1 = np.array(socreArr).astype('str')
            myThirdName = "results/DiscrepancyDistance_Estimation_CIFAR100.txt"
            f = open(myThirdName, "w", encoding="utf-8")
            for i in range(np.shape(arr1)[0]):
                f.writelines(arr1[i])
                f.writelines('\n')
            f.flush()
            f.close()


infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
# infoMultiGAN.test()
