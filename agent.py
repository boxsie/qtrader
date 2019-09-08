import os
import tensorflow as tf

from model import Model
from memory import Memory
from trainer import Trainer

class Agent:
    def __init__(self, name, num_states, num_actions, batch_size, max_memory, save_cnt, model_path):
        self._name = name
        self._batch_size = batch_size
        self._save_cnt = save_cnt
        self._model_path = model_path
        self._full_path = os.path.join(self._model_path, self._name)
        self._model = Model(num_states=num_states, num_actions=num_actions)
        self._memory = Memory(max_memory=max_memory)

        if not os.path.exists(self._model_path):
            os.mkdir(self._model_path)
        if not os.path.exists(self._full_path):
            os.mkdir(self._full_path)

    def train(self, broker, max_eps=1.0, min_eps=0.1, decay=0.1, gamma=0.5, learning_rate=1e-3):
        cnt = 0
        with tf.Session() as sess:
            trainer = self._create_trainer(sess, broker, max_eps, min_eps, decay, gamma)
            var_init = self._model.prepare(learning_rate)
            sess.run(var_init)

            if len(os.listdir(self._full_path) ) > 0:
                self.load_model(sess, self._name)

            while True:
                if trainer.train():
                    break
                cnt += 1

                if cnt % self._save_cnt == 0:
                    self.save_model(sess, self._name)

    def _create_trainer(self, sess, broker, max_eps, min_eps, decay, gamma):
        return Trainer(
            sess=sess,
            env=broker,
            model=self._model,
            memory=self._memory,
            batch_size=self._batch_size,
            max_eps=max_eps,
            min_eps=min_eps,
            decay=decay,
            gamma=gamma)

    def save_model(self, sess, model_name):
        print('')
        self._model.save(sess, self._full_path, model_name)
        self._memory.save(self._full_path, model_name)

    def load_model(self, sess, model_name):
        print('')
        self._model.load(sess, self._full_path, model_name)
        self._memory.load(self._full_path, model_name)
