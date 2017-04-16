import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

learning_rate = 0.001
training_iters = 1000
batch_size = 50
display_step = 10

dropout = 0.5
num_classes = 10

keep_prob = tf.placehoder(tf.float32)

def convnet(images, _dropout):
    parameters = [] 
    # conv1
    with tf.name_scope('conv1') as scope: 
        wb = tf.sqrt(6/(228,000+82600)) 
        kernel = tf.Variable(tf.random_uniform([5, 7, 7, 3, 4], minval=-wb, maxval=wb, 
                             dtype=tf.float32), name='weights')
        conv = tf.nn.conv3d(images, kernel, [1, 1, 1, 1, 1], padding='VALID')                                            	
        biases = tf.Variable(tf.constant(1.0, shape=[4], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]
        
    # pool1
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')
 
    # conv2
    with tf.name_scope('conv2') as scope:  
        wb = tf.sqrt(6/(82600+12960)) 
        kernel = tf.Variable(tf.random_uniform([3, 5, 5, 4, 8], minval=-wb, maxval=wb, 
                             dtype=tf.float32), name='weights')
        conv = tf.nn.conv3d(pool1, kernel, [1, 1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(1.0, shape=[8], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]
            	
    # pool2
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID')
        
    # conv3
    with tf.name_scope('conv3') as scope:  
    	  wb = tf.sqrt(6/(12960+8448)) 
        kernel = tf.Variable(tf.random_uniform([3, 5, 5, 8, 32], minval=-wb, maxval=wb, 
                             dtype=tf.float32), name='weights')
        conv = tf.nn.conv3d(pool2, kernel, [1, 1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(1.0, shape=[32], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]
            
    # pool3
    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 1, 2, 1, 1], strides=[1, 1, 2, 1, 1], padding='VALID')
        
    # conv4
    with tf.name_scope('conv4') as scope:  
        wb = tf.sqrt(6/(8448+768)) 
        kernel = tf.Variable(tf.random_uniform([3, 5, 3, 32, 64], minval=-wb, maxval=wb, 
                             dtype=tf.float32), name='weights')
        conv = tf.nn.conv3d(pool3, kernel, [1, 1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]
            
    # pool4
    pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='VALID')
        

    #fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(self.pool4.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 512],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')                     
        fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        pool4_flat = tf.reshape(self.pool4, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool4_flat, fc1w), fc1b)
        fc1a = tf.nn.relu(fc1l)
        fc1 = tf.nn.dropout(fc1a, _dropout)
        parameters += [fc1w, fc1b]    
            
    # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([512, 256],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        fc2a = tf.nn.relu(fc2l)
        fc2 = tf.nn.dropout(fc2a, _dropout)
        parameters += [fc2w, fc2b]   
            
        # softmax
    with tf.name_scope('sfmax') as scope:
        sfw = tf.Variable(tf.truncated_normal([256, num_classes],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
        sfb = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                             trainable=True, name='biases')
        sfmax = tf.nn.bias_add(tf.matmul(self.fc2, sfw), sfb)
        parameters += [sfw, sfb]
    return sfmax
            
def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')

    return tf.reduce_mean(cross_entropy, name='xentropy_mean')
                

if __name__ == '__main__':
	  imgs = tf.placeholder(tf.float32, [None, 32, 125, 57, 3]
	  pred = convnet(imgs, keep_prob)
	  y = tf.placeholder(tf.float32, [None, num_classes])
	  # define loss
	  cost = loss(pred, y)
	  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	  
	  # evaluate
	  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	  init = tf.global_variables_initializer()
	  sess.run(init)
	  step = 1
	  while step*batch_size < training_iters:
	  	  baych_xs, batch_ys = gesture.next_batch(batch_size)
	  	  sess.run(optimizer, feed_dict={imgs: batch_xs, y: batch_ys, keep_prob: dropout})
	  	  if step % display_step == 0:
	  	  	  acc = sess.run(accuracy, feed_dict={img: batch_xs, y: batch_ys, keep_prob: 1.})
	  	  	  loss = sess.run(cost, feed_dict={img: batch_xs, y: batch_ys, keep_prob: 1.})
