# autoencoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
import glob
import random
import cv2


#
# 入力層のユニット数 = 784
# 最初の中間層のユニット数 = 100 # 2つめの中間層のユニット数 = 50 # 3つめの中間層のユニット数 = 25 # 4つめの中間層のユニット数 = 12 # 5つめの中間層のユニット数 = 5 # 6つめの中間層のユニット数 = 5 num_units1 = 100
num_units2 = 50
num_units3 = 25
num_units4 = 12
num_units5 = 5
num_units6 = 5
num_input = 784
lr = 0.001
batch_size = 64
num_class = 10
num_epoch = 100
PathCKPT = './AEout/logs/model.ckpt'

class TL_MNIST:
    def __init__ (self,):

        self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, num_class], name='y')
        # 28x28x1 14x14x32 7x7x64
        conv1 = tf.layers.conv2d(self.X, 32, 5, strides=(2,2), padding = 'SAME', activation=tf.nn.relu, name='conv1', trainable=False)
        #bn1 = tf.layers.batch_normalization(conv1, training=self.is_training)
        #pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, strides=(2,2), padding='SAME', activation=tf.nn.relu, name='conv2', trainable=False)
        #bn2 = tf.layers.batch_normalization(conv2, training=self.is_training)
        #pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # fc
        flat = tf.layers.flatten(conv2)
        fc1 = tf.layers.dense(flat, 1024)
        self.logits = tf.layers.dense(fc1, num_class)


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
            gts[i, label]=1.

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
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y, 1),
                                                                             logits=self.logits))
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


        # tensorboard
        summary_loss = tf.summary.scalar('loss', loss)
        now = datetime.now()
        logdir_base = './TLout/logs/'
        logdir = logdir_base + now.strftime('%Y%m%d-%H%M%S') + '/'

        # data
        train_data = self.load_images('JPEGImages/train/')

        # restore
        reusevar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv[12]')
        rvdict = dict([(var.op.name, var) for var in reusevar])


        saver = tf.train.Saver(rvdict)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, PathCKPT)

            global_step = 0
            for i in range(num_epoch):
                last = 0
                step_size = math.ceil(60000/batch_size)
                for step in range(step_size):
                    data, labels, last = self.get_batch(train_data, last)
                    _, sloss = sess.run([train_op, summary_loss], feed_dict={self.X:data, self.y:labels})

                    if global_step % 200 == 0:
                        currentloss = sess.run([loss], feed_dict={self.X:data, self.y: labels})
                        print(currentloss)
                        #print(sess.run(reusevar))
                        #print(data.shape, labels.shape)
                        #print(np.argmax(labels[0]))
                        #saveimg = np.vstack((data[0].reshape(28,28)*255, aeimg[0].reshape(28,28)*255))
                        #cv2.imwrite('./AEout/{0}_{1}.jpg'.format(np.argmax(labels[0]), global_step), saveimg)
                        #cv2.imwrite('./AEout/hogehoge.jpg', aeimg[0].reshape(28,28))
                        #writer.add_summary(sloss, global_step)
                    global_step += 1


if __name__ == '__main__':
    TL_MNIST().trainer()


"""

# 自己符号化関数
def auto_encoding(x, w_enc, b_enc, w_dec, b_dec):
    encoded = tf.nn.relu(tf.matmul(x, w_enc) + b_enc)
    decoded = tf.matmul(encoded, w_dec) + b_dec
    return decoded


x = tf.placeholder(tf.float32, [None, 784])

w_enc = tf.Variable(tf.random_normal([784, num_units1], stddev=0.01))
b_enc = tf.Variable(tf.zeros([num_units1]))

w_dec = tf.Variable(tf.random_normal([num_units1, 784], stddev=0.01))
b_dec = tf.Variable(tf.zeros([784]))

# 初回の自己符号化器の誤差関数
decoded = auto_encoding(x, w_enc, b_enc, w_dec, b_dec)

loss = tf.reduce_sum(tf.square(x - decoded))
train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(3000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs})
    if i % 100 == 0:
        loss_val = sess.run(loss,
                            feed_dict={x: mnist.train.images})
        print('Step: %d, Loss: %f' % (i, loss_val))
# 学習した重み
w_dec_p, b_dec_p, w_enc_p, b_enc_p = sess.run([w_dec, b_dec, w_enc, b_enc])


second_x = tf.nn.relu(tf.matmul(x, w_enc_p) + b_enc_p)
decoded = auto_encoding(second_x, w2_enc, b2_enc, w2_dec, b2_dec)

# 事前学習した重みを使ってのMNIST分類精度
x = tf.placeholder(tf.float32, [None, 784])

w1_n = tf.Variable(w_enc_p)
b1_n = tf.Variable(b_enc_p)
hidden1 = tf.nn.relu(tf.matmul(x, w1_n) + b1_n)

w2_n = tf.Variable(w2_enc_p)
b2_n = tf.Variable(b2_enc_p)
hidden2 = tf.nn.relu(tf.matmul(hidden1, w2_n) + b2_n)

w3_n = tf.Variable(w3_enc_p)
b3_n = tf.Variable(b3_enc_p)
hidden3 = tf.nn.relu(tf.matmul(hidden2, w3_n) + b3_n)

w4_n = tf.Variable(w4_enc_p)
b4_n = tf.Variable(b4_enc_p)
hidden4 = tf.nn.relu(tf.matmul(hidden3, w4_n) + b4_n)

w5_n = tf.Variable(w5_enc_p)
b5_n = tf.Variable(b5_enc_p)
hidden5 = tf.nn.relu(tf.matmul(hidden4, w5_n) + b5_n)

w6_n = tf.Variable(w6_enc_p)
b6_n = tf.Variable(b6_enc_p)
hidden6 = tf.nn.relu(tf.matmul(hidden5, w6_n) + b6_n)

# 一番上位にランダムに初期化した重みの層をつける
w0_n = tf.Variable(tf.random_normal([num_units6, 10], stddev=0.01))
b0_n = tf.Variable(tf.random_normal([10]))
p = tf.nn.softmax(tf.matmul(hidden6, w0_n) + b0_n)

# 誤差関数の設定と学習
t = tf.placeholder(tf.float32, [None, 10])

loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# 30,000 step実行
i = 0
accuracy_list_stacked = []
index_list = []
for _ in range(30000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                                     feed_dict={x: mnist.test.images, t: mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f'
              % (i, loss_val, acc_val))
        accuracy_list_stacked.append(acc_val)
        index_list.append(i)

# 通常のネットワークと事前学習ネットワークの精度比較
plt.figure(figsize=(10, 5))
plt.xlabel('times (* 100)')
plt.ylabel('accuracy')

plt.plot(accuracy_list, linewidth=1, color="red", label="normal")
plt.plot(accuracy_list_stacked, linewidth=1, color="blue", label="pretrained")
plt.legend(loc="lower right")
"""