
import cv2
import numpy as np
import scipy.ndimage
import tensorflow as tf
import time

def main():

    hrImg = 255 - cv2.cvtColor(cv2.imread('./hour.png') ,  cv2.COLOR_RGB2GRAY )
    mnImg = 255 - cv2.cvtColor(cv2.imread('./minute.png'), cv2.COLOR_RGB2GRAY )
    
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
    
    imgBatch, unaryHr, unaryMn = makeClockBatch(5, hrImg, mnImg)
    
    np.set_printoptions(threshold=np.nan) 
   
    
    def inference(batchSize = 10, printClocks = False):
        numRight = 0
        for t in range(batchSize):
            hr = np.random.randint(0,12)
            mn = np.random.randint(0,12) * 5
            
            testClock = [makeClock(hr, mn, hrImg, mnImg)]
            
            outHr = np.argmax(sess.run(fc_out_hr, feed_dict={x:testClock}))
            outMn = np.argmax(sess.run(fc_out_mn, feed_dict={x:testClock}))
            outMn *= 5 
            
            if(printClocks):
                print ("Is it %02d:%02d? - %02d:%02d"%(hr,mn, outHr, outMn))
   
            if (hr == outHr and mn == outMn):
                numRight += 1
    
        return numRight

    batchSize = 32
    numBatches= 25
    numEpochs = 5

    for e in range(numEpochs):
        startTime = time.clock()
        for b in range(numBatches):
            imgBatch, unaryHr, unaryMn = makeClockBatch(batchSize, hrImg, mnImg)
            train_step.run(feed_dict={x: imgBatch, y_hr: unaryHr, y_mn: unaryMn})
        
        imgBatch, unaryHr, unaryMn = makeClockBatch(10, hrImg, mnImg)
        epochTime = time.clock()-startTime
        trainAccuracy = inference(20);
        print("epoch %02d, training accuracy %g%%:\t%gs" %(e, 100*trainAccuracy/20, epochTime))
        if( trainAccuracy > 0.9 and e > 3):
            break

    print("running inference, Correct predictions: ", inference(10, True))

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
  

def makeClock(hr, mn, hrImg, mnImg):
    rot = scipy.ndimage.interpolation.rotate(mnImg, -mn*6, reshape=False)
    rot +=  scipy.ndimage.interpolation.rotate(hrImg, -(hr*30.0 + mn/2.0), reshape=False)
    rot = rot / 255
    return rot

def makeClockBatch(batchSize, hrImg, mnImg):
    
    assert hrImg.shape == mnImg.shape, "Image size mismatch in hrImg ang mnImg"
    batch = np.empty([batchSize, hrImg.shape[0], hrImg.shape[1]])
    hrs = np.zeros([batchSize, 12])
    mns = np.zeros([batchSize, 12])
    
    for i in range(batchSize):
        hour = np.random.randint(0,12) # because 0 == 12 on clock.
        mins = np.random.randint(0,12)
        #print("time:%02d:%02d"%(hour,mins))
        hrs[i][hour] = 1
        mns[i][mins] = 1
        batch[i] = makeClock(hour, mins*5, hrImg, mnImg)
        
    return batch, hrs, mns


if __name__== "__main__":
   main()