import os
import tensorflow as tf

class Model:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self._logits = None
        self._var_init = None
        self._optimiser = None
        self._saver = None
        self._states = tf.compat.v1.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self._qsa = tf.compat.v1.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        fc1 = tf.layers.dense(self._states, 64, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc3, self.num_actions)

    def prepare(self, learning_rate):
        loss = tf.compat.v1.losses.mean_squared_error(self._qsa, self._logits)
        self._optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        self._var_init = tf.compat.v1.global_variables_initializer()
        self._saver = tf.compat.v1.train.Saver()
        return self._var_init

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimiser, feed_dict={self._states: x_batch, self._qsa: y_batch})

    def save(self, sess, path, filename):
        full = os.path.join(path, f'{filename}.ckpt')
        save_path = self._saver.save(sess, f'{full}')
        print(f'Model saved in path: {full}')

    def load(self, sess, path, filename):
        full = os.path.join(path, f'{filename}.ckpt')
        self._saver.restore(sess, f'{full}')
        print(f'Model loaded from path: {full}')