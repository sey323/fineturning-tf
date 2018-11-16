import sys , os
import json
import os
import sys
import tensorflow as tf
from datetime import datetime


class vgg19( object ):
    '''
    VGG19の転移学習
    '''
    def __init__(   self,
                    input_size,
                    channel,
                    labelSize,
                    nBatch,
                    drop_prob = 0.5 ,
                    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.1) ,
                    saveFolder = "models/CNN"):
        self.input_size = input_size
        self.channel = channel
        self.labelSize = labelSize
        self.nBatch = nBatch
        self.drop_prob = drop_prob

        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.saveFolder = saveFolder + "/" + now
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"images")):
            os.makedirs(os.path.join(self.saveFolder))
            param_file = open(self.saveFolder+'/param.json','w')
            # パラメータを保存
            json.dump(self.__dict__,param_file,indent=2)

        # GPUの設定
        self.gpu_config = tf.ConfigProto( gpu_options = gpu_config )


    def _conv_variable( self, weight_shape , name="conv" ):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias


    def _fc_variable( self, weight_shape , name="fc" ):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = ( input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias


    def conv2d( self, x , stride , filter_size , i ,padding = 'SAME',BatchNorm = False):
        '''
        畳み込み(Convolution)
        ----
        @x          :input
        @filter_size[0],[1] : Conv filter(width,height)
        @filter_size[2]     : input_shape(直前のoutputの次元数と合わせる)
        @filter_size[3]     : output_shape(出力する次元数)
        '''
        conv_w,conv_b = _conv_variable([ filter_size[0], filter_size[1], filter_size[2] , filter_size[3] ],name="conv{0}".format(i))
        conv =  tf.nn.conv2d( x,                                # 入力
                            conv_w,                             # 畳み込みフィルタ
                            strides = [1, stride, stride, 1],   # ストライド
                            padding = padding) + conv_b
        if BatchNorm:
            conv = self(conv)
        return conv


    def fc( self, input , output , i = None,BatchNorm = False):
        '''
        全結合層(Fully Connection)
        ----
        @input      : input
        @output     : output shape (example:classes num)
        @nBatch     : Batch size
        '''
        _, n_h = [int(x) for x in input.get_shape()]
        d_fc_w, d_fc_b = self._fc_variable([n_h,output],name="fc{0}".format(i))
        fc = tf.matmul( input , d_fc_w) + d_fc_b
        return fc


    def flatten( self, input , batch_num , name = None):
        '''
        flatten(平滑化層)
        -----
        input      : input shape
        output     : output shape
        '''
        _, n_h, n_w, n_f = [x for x in input.get_shape()]
        output = tf.reshape(input ,[batch_num,int(n_h)*int(n_w)*int(n_f)])
        return output


    def batchNorm( self, x , decay=0.9 , updates_collections=None, epsilon=1e-5, scale=True, is_training=True, scope=None):
        '''
        BatchNorm
        x               : input
        '''
        return tf.contrib.layers.batch_norm(x,
                                            decay=decay,
                                            updates_collections=updates_collections,
                                            epsilon=epsilon,
                                            scale=scale,
                                            is_training=is_training,
                                            scope = scope )


    def cross_entropy( self , x , labels, nBatch , name = '' ):
        '''
        SoftMaxの計算誤差
        '''
        with tf.variable_scope('cross_entropy_softmax'+name):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=x, name='cross_entropy_per_example')
            cross_entropy_softmax = tf.reduce_mean(cross_entropy, name = 'cross_entropy_softmax')
        return cross_entropy_softmax


    def vgg19(self , input , reuse = False , freeze = True,name =''):
        '''
        VGG19
        '''
        print('[SETUPMODEL]\tVGG19')
        with tf.variable_scope("VGG19" + name , reuse=reuse) as scope:
            if reuse: scope.reuse_variables()
            with tf.variable_scope("VGG19"):
                vgg19     = tf.keras.applications.VGG19(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=(self.input_size[0], self.input_size[1], self.channel),
                                                        input_tensor=input
                                                        )
                if freeze:# 下位層のみ学習する場合
                    for vgg_layer in vgg19.layers:
                        vgg_layer.trainable = False

                # 最後の層の出力を得る．
                vgg19_out = tf.identity(vgg19.layers[-1].output, name='output')

            '''
            自作でFC層のみ学習する．
            '''
            with tf.variable_scope("FC{0}".format(1)):
                flat       = self.flatten(vgg19_out,self.nBatch)
                fc1 = self.fc( flat , 4096 , "1",BatchNorm = False)
                fc2 = self.fc( fc1 ,  4096 , "2",BatchNorm = False)
                fc3 = self.fc( fc2 ,  1000 , "3",BatchNorm = False)
                output = self.fc( fc3 , self.labelSize , "out",BatchNorm = False)

        return output


    def buildModel(self):
        '''
        計算式の定義
        '''
        # Model
        self.x = tf.placeholder(tf.float32, [ None , self.input_size[0] , self.input_size[1] , 3] , name = "train_image")
        self.labels = tf.placeholder( tf.float32 , [ None , self.labelSize ], name = "train_label")
        self._drop_prob = tf.placeholder(tf.float32)

        # モデルの構築
        self.vgg19_model = self.vgg19(self.x)

        # 損失関数の計算
        self.loss = self.cross_entropy(x=self.vgg19_model,labels=self.labels,nBatch=self.nBatch,name="cross_entropy")
        self.vgg_optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # Accuracy
        self.accuracy = self.accuracy(self.vgg19_model, self.labels)

        self.sess = tf.Session(config=self.gpu_config)

        # summary
        tf.summary.scalar( "loss" , self.loss )
        tf.summary.scalar( "accuracy" , self.accuracy )

        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)


    def fit(self, train_batch, test_batch , max_epoch=200):
        '''
        モデルの学習
        '''
        # 初期化
        self.buildModel()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        step = 0
        while train_batch.getEpoch()<max_epoch:
            train_image_batch , train_label_batch = train_batch.getBatch( self.nBatch )
            test_image_batch , test_label_batch = test_batch.getBatch( self.nBatch )

            _, summary = self.sess.run( [self.vgg_optimizer,self.summary] , feed_dict={self.x: train_image_batch, self.labels: train_label_batch,self._drop_prob:self.drop_prob })

            if step>0 and step%10==0:
                self.writer.add_summary(summary , step )

            if step%100==0:
                # 毎ステップ、学習データに対する正答率と損失関数を記録
                loss_ = self.loss.eval(session=self.sess, feed_dict={ self.x: test_image_batch, self.labels: test_label_batch ,self._drop_prob:1.0 })
                accuracy_ = self.accuracy.eval(session=self.sess,feed_dict={ self.x:test_image_batch, self.labels: test_label_batch,self._drop_prob:1.0})
                print("Epoch %d :step %d, test accuracy %g loss %g"%(train_batch.getEpoch(),step, accuracy_ , loss_ ))
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)

            step+=1


    def accuracy( self , y , t , name = 'accuracy'):
        '''
        モデルの性能の評価
        '''
        with tf.variable_scope(name):
            correct_prediction = tf.equal( tf.argmax( y , 1 ) , tf.argmax(t ,1 ))
            accuracy = tf.reduce_mean( tf.cast(correct_prediction , tf.float32))
        return accuracy
