import math
import random
import numpy as np
import pandas as pd

class Trainer:
    def __init__(self, sess, env, model, memory, batch_size, max_eps, min_eps, decay, gamma):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._batch_size = batch_size
        self._state = np.array([])
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._gamma = gamma
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._total_reward = 0

    def train(self):
        action = self._choose_action(self._state) if self._state.any() else 0
        is_complete, next_state, reward, stats = self._env.update(action)

        if not self._state.any():
            self._state = next_state
        else:
            self._memory.add_sample((self._state, action, reward, next_state))
            self._replay()

            self._steps += 1
            self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._decay * self._steps)

            self._state = next_state
            self._total_reward += reward

            if is_complete:
                self._reward_store.append(self._total_reward)

        return is_complete, self._state, reward, stats

    def _choose_action(self, state):
        rnd = random.random()
        if rnd < self._eps:
            return random.randint(0, self._env.num_actions - 1)
        return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states) if val[3] is None else val[3]) for val in batch])

        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)

        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))

        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]

            # get the current q values for all actions in state
            current_q = q_s_a[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q

        self._model.train_batch(self._sess, x, y)