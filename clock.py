
import cv2
import numpy as np
import scipy.ndimage
import tensorflow as tf
import time

class ClockMaker:

    def __init__(self, hrImgin, mnImgin):
        assert hrImgin.shape == mnImgin.shape, "Image size mismatch in hrImg ang mnImg"

        self.hrImg = hrImgin / 255
        self.mnImg = np.zeros([12, mnImgin.shape[0], mnImgin.shape[1]])
        for i in range(12):
            self.mnImg[i] = scipy.ndimage.interpolation.rotate(mnImgin, -i*30, reshape=False) / 255


    def makeClock(self, hr, mn):
        mn = int(mn/5) # multiples of 5 only
        return self.mnImg[mn] + scipy.ndimage.interpolation.rotate(self.hrImg, -hr*30.0 - mn/2.0,reshape=False)
        
    def makeClockBatch(self, batchSize):
        
        batch = np.empty([batchSize, self.hrImg.shape[0], self.hrImg.shape[1]])
        hrs = np.zeros([batchSize, 12])
        mns = np.zeros([batchSize, 12])
        
        for i in range(batchSize):
            hour = np.random.randint(0,12)
            mins = np.random.randint(0,12)
            hrs[i][hour] = 1
            mns[i][mins] = 1
            batch[i] = self.makeClock(hour, mins * 5)
            
        return batch, hrs, mns    
    
    def show(self, hr, mn):
        clockFace = self.makeClock(hr,mn)
        cv2.imshow( "%02d:%02d"%(hr,mn), clockFace)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():

    clockMaker = ClockMaker(255 - cv2.cvtColor(cv2.imread('./hour.png') , cv2.COLOR_RGB2GRAY ), \
                            255 - cv2.cvtColor(cv2.imread('./minute.png'), cv2.COLOR_RGB2GRAY ), )

    x = tf.placeholder(tf.float32, [None, 128, 128], name ='inputImg')

    x_image = tf.reshape(x, [-1,128,128,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, weight_variable([5, 5, 1, 8])) + bias_variable([8]), name='conv1') # 128x128x8
    h_pool1 = max_pool_2x2(h_conv1)     # 64x64x8
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_variable([5, 5, 8, 16])) + bias_variable([16]), name='conv2') # 64x64x16
    h_pool2 = max_pool_2x2(h_conv2)  # 32x32x16

    h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*16])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_variable([32* 32 * 16, 128])) + bias_variable([128]), name='FC1')
    
    fc_out_hr = tf.nn.relu( tf.matmul(h_fc1, weight_variable([128, 12])) + bias_variable([12]), name='FC2Hr')
    fc_out_mn = tf.nn.relu( tf.matmul(h_fc1, weight_variable([128, 12])) + bias_variable([12]), name='FC2Mn')

    y_hr = tf.placeholder(tf.float32, [None, 12], name='y_hr')
    y_mn = tf.placeholder(tf.float32, [None, 12], name='y_mn')
        
    sess = tf.InteractiveSession()
    
    total_out = tf.add( tf.nn.softmax_cross_entropy_with_logits(labels=y_hr, logits=fc_out_hr) ,\
                        tf.nn.softmax_cross_entropy_with_logits(labels=y_mn, logits=fc_out_mn) )
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(tf.reduce_mean(total_out))
    
    sess.run(tf.global_variables_initializer())    

    def inference(batchSize = 10, printClocks = False):
        numRight = 0
        rmsTickError = 0 
        for t in range(batchSize):
            hr = np.random.randint(0,12)
            mn = np.random.randint(0,12) * 5
            testClock = [clockMaker.makeClock(hr, mn)]
            
            outHr = np.argmax(sess.run(fc_out_hr, feed_dict={x:testClock}))
            outMn = np.argmax(sess.run(fc_out_mn, feed_dict={x:testClock})) * 5
            
            if(printClocks):
                print ("Is it %02d:%02d? - %02d:%02d"%(hr,mn, outHr, outMn))
   
            if (hr == outHr and mn == outMn):
                numRight += 1

            rmsTickError += pow( (hr - outHr)*(hr - outHr) + (mn - outMn)*(mn - outMn) , 0.5)
    
        return numRight/batchSize, rmsTickError/batchSize

    batchSize = 32
    numBatches= 20
    numEpochs = 10

    for e in range(numEpochs):
        startTime = time.clock()
        for b in range(numBatches):
            imgBatch, unaryHr, unaryMn = clockMaker.makeClockBatch(batchSize)
            train_step.run(feed_dict={x: imgBatch, y_hr: unaryHr, y_mn: unaryMn})
        
        epochTime = time.clock()-startTime
        numTestImages = 8
        trainAccuracy, rmsTickError = inference(numTestImages)
        print("epoch %02d, training accuracy: %2g%%\t| Ticks RMS Error: %2g -\t%2gs" %(e, trainAccuracy*100 , rmsTickError, epochTime))
        if( trainAccuracy > .9 and e > 3):
            break

    print("\nRunning inference")

    acc, tickRms = inference(10, True)
    print("inference, Correct predictions: ", acc*100, "% - tick error: ", tickRms )

    sess.close()    
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05, mean=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def max_pool_2x2(x, ):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__== "__main__":
   main()