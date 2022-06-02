import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from random import sample

from ImitationDataset import ImitationDataset
from ImitationNet import ImitationNet



def load_data(path):
    """
    Helper method to load the dataset
    :param path: Path to location of dataset
    :return: lists of all the state arrays
    """
    state_paths = list(path.glob('*.npy'))

    if len(state_paths) < 1:
        raise ValueError('Invalid data loaded')
    
    return np.array(state_paths)


class ImitationAgent:
    def __init__(self, val_percentage, test_num,
                 batch_size, inp, out, data_path, shuffle_data,
                 learning_rate, device):
        """
        A helper class to facilitate the training of the model
        """
        self.device = device
        self.batch_size = batch_size
        self.state_paths = load_data(data_path)
        train_split, val_split, test_split = self.make_splits(
                val_percentage, test_num, shuffle_data)
        self.train_loader = self.get_dataloader(train_split)
        self.validation_loader = self.get_dataloader(val_split)
        self.test_loader = self.get_dataloader(test_split)
        self.model = ImitationNet(inp, out)
        self.criterion = nn.MSELoss() 
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def make_splits(self, val_percentage=0.2, test_num=10, shuffle=True):
        """
        Split the data into train, validation and test datasets
        :param val_percentage: A decimal number which tells the percentage of
                data to use for validation
        :param test_num: The number of images to use for testing
        :param shuffle: Shuffle the data before making splits
        :return: tuples of splits
        """
        if shuffle:
            shuffle_idx = np.random.permutation(range(len(self.state_paths)))
            self.state_paths = self.state_paths[shuffle_idx]

        val_num = len(self.state_paths) - int(
            val_percentage * len(self.state_paths))

        train_states = self.state_paths[:val_num]

        validation_states = self.state_paths[val_num:-test_num]

        test_states = self.state_paths[-test_num:]

        return train_states, validation_states, test_states

    def get_dataloader(self, split):
        """
        Create a DataLoader for the given split
        :param split: train split, validation split or test split of the data
        :return: DataLoader
        """
        return DataLoader(ImitationDataset(split, self.device), self.batch_size, shuffle=True)