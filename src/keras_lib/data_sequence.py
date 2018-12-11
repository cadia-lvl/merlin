from keras.utils import Sequence


class UttBatchSequence(Sequence):

    def __init__(self, x_set, y_set):
        # x_set, y_set must be lists of numpy arrays
        self.x, self.y = x_set, y_set

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        batch_x = self.x[idx].reshape((1, self.x[idx].shape[0], self.x[idx].shape[1]))
        batch_y = self.y[idx].reshape((1, self.y[idx].shape[0], self.y[idx].shape[1]))

        return batch_x, batch_y
