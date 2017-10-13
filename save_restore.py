import tensorflow as tf



## save to file
'''
W = tf.Variable([[1,2,3], [4,5,6]], dtype=tf.float32, name='Weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='bias')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'mnist_data/save_data.ckpt')
    print('Save to path:', save_path)
'''

## restore variable
# redefine the same shape and same type for your variable

W = tf.Variable(tf.zeros([2,3]), dtype=tf.float32, name='Weights')
b = tf.Variable(tf.zeros([1,3]), dtype=tf.float32, name='bias')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'mnist_data/save_data.ckpt')
    print('weights:', sess.run(W))
    print('biases:', sess.run(b))
