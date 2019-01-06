# パッケージの読み込み
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import math
import glob
import random
import cv2


lr = 0.001
batch_size = 64
num_input = 784
num_class = 10
num_epoch = 100


# モデルのクラス
class CNN_MNIST:
    def __init__(self, is_training):

        self.is_training = is_training

        self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, num_class], name='y')
        conv1 = tf.layers.conv2d(self.X, 32, 5, activation=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(conv1, training=self.is_training)
        pool1 = tf.layers.max_pooling2d(bn1, 2, 2)
        conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
        bn2 = tf.layers.batch_normalization(conv2, training=self.is_training)
        pool2 = tf.layers.max_pooling2d(bn2, 2, 2)
        flat = tf.layers.flatten(pool2)
        fc1 = tf.layers.dense(flat, 1024)
        self.logits = tf.layers.dense(fc1, num_class)
        #self.prediction = tf.nn.softmax(self.logits)

    def load_images(self, dataset_path, shuffle=True):
        filepaths = glob.glob(dataset_path + '*.jp*g')
        filepaths.sort()

        imgs = np.zeros((len(filepaths), 28, 28, 1), dtype=np.float32)
        gts = np.zeros((len(filepaths), num_class), dtype=np.float32)

        for i, filepath in enumerate(filepaths):
            img = cv2.imread(filepath, 0)
            img = cv2.resize(img, (28, 28)) / 255.

            label = int(filepath.split('/')[-1].split('_')[0])
            imgs[i] = img.reshape(28,28,1)
            gts[i, label] = 1.

        inds = np.arange(len(filepaths))

        if shuffle:
            random.shuffle(inds)

        imgs = imgs[inds]
        gts = gts[inds]

        dataset = [imgs, gts]

        return dataset

    def get_batch(self, data, last):
        imgs, gts = data

        data_num = len(imgs)
        ind = last + batch_size

        if ind < data_num:
            img = imgs[last: ind]
            gt = gts[last: ind]
            last = ind
        else:
            resi = ind - data_num
            img1, gt1 = imgs[last:], gts[last:]
            img2, gt2 = imgs[:resi], gts[:resi]
            img = np.vstack((img1, img2))
            gt = np.vstack((gt1, gt2))
            last = resi

        return img, gt, last

    def trainer(self):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y,1),
                                                                             logits=self.logits)
                              )
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss)
        correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # tensorboard
        sumary_loss = tf.summary.scalar('loss', loss)
        now = datetime.now()
        logdir_base = 'logs/'
        logdir = logdir_base + now.strftime("%Y%m%d-%H%M%S") + "/"

        # data
        train_data = self.load_images('JPEGImages/train/')

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir, sess.graph)
            sess.run(tf.global_variables_initializer())

            global_step = 0
            for i in range(num_epoch):

                last = 0
                step_size = math.ceil(60000/batch_size)
                for step in range(step_size):
                    data, labels, last = self.get_batch(train_data, last)
                    _, _, sloss = sess.run([train_op, extra_update_ops, sumary_loss],
                             feed_dict={self.X: data, self.y: labels})

                    if global_step % 200 == 0:
                        currentloss, acc = sess.run([loss, accuracy], feed_dict={self.X: data, self.y: labels})
                        #sess.run([loss], feed_dict={self.X: data, self.y: labels})
                        print(currentloss, acc)
                        writer.add_summary(sloss, global_step)
                    global_step += 1


if __name__ == '__main__':
    CNN_MNIST(is_training=True).trainer()


# # MNISTデータの読み込み
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# # 入力層のユニット数 = 784
# # 中間層のユニット数 = 100 # として学習を行う
# num_units = 100
# # 入力x
# x = tf.placeholder(tf.float32, [None, 784])
# # 第1層の重み、バイアス
# # 分散0.01のガウス分布で初期化
# w_enc = tf.Variable(tf.random_normal([784, num_units], stddev=0.01))
# b_enc = tf.Variable(tf.zeros([num_units]))
#
# # 第2層の重み、バイアス
# # 分散0.01のガウス分布で初期化
# w_dec = tf.Variable(tf.random_normal([num_units, 784], stddev=0.01))
# b_dec = tf.Variable(tf.zeros([784]))
# # 中間層の活性化関数は正規化線形関数
# encoded = tf.nn.relu(tf.matmul(x, w_enc) + b_enc)
# # 出力層は恒等写像
# decoded = tf.matmul(encoded, w_dec) + b_dec
# # 誤差関数は2乗誤差の総和
# # かつ λ = 0.1の重み減衰を行う
# lambda2 = 0.1
# l2_norm = tf.nn.l2_loss(w_enc) + tf.nn.l2_loss(w_dec)
# loss = tf.reduce_sum(tf.square(x - decoded)) + lambda2 * l2_norm
# # 学習係数 0.001, のAdamのアルゴリズムをSDGに用いる
# train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# # セッションを用意し、変数を初期化する
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# # バッチサイズ100の確率的勾配降下法を1000回行う
# i = 0
# for _ in range(1000):
#     i += 1
#     batch_xs, batch_ts = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs})
#     if i % 100 == 0:
#         loss_val = sess.run(loss,
#                             feed_dict={x: mnist.train.images})
#         print('Step: %d, Loss: %f' % (i, loss_val))
# # 学習した重みとバイアスを抽出
# w_dec_p, b_dec_p, w_enc_p, b_enc_p = sess.run([w_dec, b_dec, w_enc, b_enc])
# # 訓練画像
# x_train = mnist.train.images
#
# # 自己符号化した画像の表示
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):  # もともとの画像
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_train[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# # 自己符号化した画像
# ax = plt.subplot(2, n, i + 1 + n)
# encoded_img = np.dot(x_train[i].reshape(-1, 784), w_enc_p) + b_enc_p
# decoded_img = np.dot(encoded_img.reshape(-1, 100), w_dec_p) + b_dec_p
#
# plt.imshow(decoded_img.reshape(28, 28))
# plt.gray()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
