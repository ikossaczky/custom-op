# Apache 2.0 license
import tensorflow as tf
import tensorflow_inner_product

x = tf.placeholder(tf.float32, shape=(3))
A = tf.constant([[1., 2., 3.], [4., 5., 6.]])

x_column = x[:,None]
F = tf.matmul(A, x_column)
G = tensorflow_inner_product.inner_product(x_column, A)

dsumF_dx = tf.gradients(F, x)
dsumG_dx = tf.gradients(G, x)
dsumF_dA = tf.gradients(F, A)
dsumG_dA = tf.gradients(G, A)

with tf.Session('') as sess:
    feed_dict={x:[1.,2.,3.]}
    print("F: ", sess.run(F, feed_dict))
    print("dsumF_dx: ", sess.run(dsumF_dx, feed_dict))
    print("dsumF_dA: ", sess.run(dsumF_dA, feed_dict))
    print("G: ", sess.run(G, feed_dict))
    print("dsumG_dx: ", sess.run(dsumG_dx, feed_dict))
    print("dsumG_dA: ", sess.run(dsumG_dA, feed_dict))
