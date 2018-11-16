import sys , os
import math,cv2
import tensorflow as tf
sys.path.append('./util')
import imload
from batchgen import *
from model import vgg19


'''
モデルの学習
'''
def main(FLAGS):
    img_path = os.getenv("DATASET_FOLDER", "dataset")
    save_path = os.getenv("SAVE_FOLDER", "results")

    # パラメータの取得
    img_path = FLAGS.img_path
    resize = [ FLAGS.resize , FLAGS.resize ]
    train_num = FLAGS.train_num
    test_num = FLAGS.test_num
    gray = FLAGS.gray
    channel = 1 if gray else 3

    max_epoch = FLAGS.max_epoch
    nBatch = FLAGS.batch_size

    config = tf.GPUOptions() # specify GPU number)

    # データのローディング
    train,train_label,var_train,var_label = imload.make( img_path,img_size=resize[0],train_num = train_num,test_num=test_num,gray=gray)
    train_batch = batchgen( train ,label = train_label,channel = channel)
    test_batch = batchgen( var_train ,label = var_label,channel = channel)
    labelSize = len(train_label[0])

    print('[LAODING]\tVGGモデル')

    #モデル設定
    model = vgg19(
                input_size=resize,
                channel =channel,
                labelSize=labelSize,
                nBatch = nBatch,
                saveFolder=save_path)


    #モデル学習
    model.fit(train_batch,test_batch ,max_epoch = max_epoch)


if __name__=="__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # 読み込む画像周り
    flags.DEFINE_string('img_path', '', 'Directory to put the training data.')
    flags.DEFINE_integer('resize', 64, 'Size of Image.')
    flags.DEFINE_integer('train_num', 0, 'Loading Images Num.')
    flags.DEFINE_integer('test_num', 0, 'Loading test Num.')
    flags.DEFINE_boolean('gray', False, 'Convert Gray Scale?')

    # CNNのパラメータ
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('max_epoch', 200, 'Number of steps to run trainer.')
    flags.DEFINE_integer('batch_size', 64, 'Batch size.  ''Must divide evenly into the dataset sizes.')

    main(FLAGS)
