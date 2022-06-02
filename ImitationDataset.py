from numpy import load, array
from torch import from_numpy
from torch.utils.data import Dataset


class ImitationDataset(Dataset):
    def __init__(self, state_paths, device):
        self.state_paths = state_paths
        self.device = device

    def __len__(self):
        return len(self.state_paths)

    def __getitem__(self, idx):
        states = load(self.state_paths[idx])
        state = states[0]
        action = array([[states[1, 0, 4]]])
        state = from_numpy(state).float().to(self.device)
        action = from_numpy(action).float().to(self.device)
        # action = action * 90
        return state, action