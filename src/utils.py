import numpy as np
from tensorflow.keras.utils import Sequence
from src.preprocess import load_image


class DicomSequence(Sequence):
    def __init__(self, filepaths, labels, batch_size=8, target_size=(224,224), shuffle=True):
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.indexes = np.arange(len(self.filepaths))
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = [load_image(self.filepaths[i], target_size=self.target_size) for i in batch_idx]
        batch_y = [self.labels[i] for i in batch_idx]
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
