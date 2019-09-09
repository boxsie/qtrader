import os.path
import random
import pickle

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        return random.sample(self._samples, no_samples)

    def save(self, path, filename):
        full = os.path.join(path, f'{filename}.pkl')
        with open(f'{full}', 'wb') as fp:
            pickle.dump(self._samples, fp)
        print(f'Memory saved in path: {full}')

    def load(self, path, filename):
        full = os.path.join(path, f'{filename}.pkl')
        with open(full, 'rb') as fp:
            self._samples = pickle.load(fp)
        print(f'Memory loaded from path: {full}')