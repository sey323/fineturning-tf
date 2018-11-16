import numpy as np
import math

'''
バッチを生成するクラス．
'''
class batchgen(object):
    def __init__( self , image , label , size = None ,channel = 3 , labeltype = 'onehot'):
        if size is None:
            width = int(math.sqrt(len(image[0])/channel))
            height = int(math.sqrt(len(image[0])/channel))
            size = [ width , height ]

        self.size = size
        self.channel = channel

        self.image = np.array(self._image2tensor(image))
        self.label = label

        # 乱数のindex
        self.idx = None

        # バッチのスタート位置
        self.start_idx = 0
        self.end_idx = 0

        self.epoch = 0

        # onehotじゃない時の処理
        if labeltype == 'onehot':
            self.label = np.array(self.label)
        else:
            self._onehot2array()

    '''
    画像をtensorflowで学習可能な形式に変換．
    '''
    def _image2tensor( self , img ):
        tensor = np.reshape(img,[len(img),self.size[0],self.size[1],self.channel])
        return tensor

    '''
    one-hot表現された配列を，ラベル付き配列に戻す．
    [ 0 , 0 , 1 ]→[ 3 ]
    '''
    def _onehot2array( self ):
        onehot = self.label.tolist()
        array = []
        for o in onehot:
            index = o.index( 1 )
            array.append( index )
        self.label = np.array(array)


    '''
    画像とラベルをバッチサイズ分取得する．
    type@ tensor : numpy.ndarray
    type@ label : numpy.ndarray
    '''
    def getBatch( self , nBatch , idx = None):
        self.end_idx = self.start_idx + nBatch
        # 終了位置か判定
        if self.end_idx > len(self.image):
            self.start_idx = 0
            self.end_idx = nBatch
            self.epoch += 1
            if idx is None:
                self.shuffle()
            else:
                self.shuffle(idx)

        # normalized to -0.5 ~ +0.5
        tensor,label = self.image[self.start_idx:self.end_idx],self.label[self.start_idx:self.end_idx]
        tensor = (tensor-0.5)/1.0

        self.start_idx += nBatch
        return tensor , label


    '''
    ラベルと画像を同じ乱数で並び替える．
    '''
    def shuffle(self,idx = None):
        # 乱数シードが設定されていない時
        if idx is None:
            idx = np.random.permutation(len(self.image))

        self.image = self.image[idx]
        self.label = self.label[idx]

        self.idx = idx


    '''
    シャッフルした乱数のindexを返す．
    '''
    def getIndex( self ):
        return self.idx


    '''
    バッチ内の画像のサイズを返す
    '''
    def getNum(self):
        return len(self.image)

    '''
    Epochを返す．
    '''
    def getEpoch(self):
        return self.epoch
