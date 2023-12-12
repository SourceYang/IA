from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(self, input_data, out1, out2, out3):
        self.input_data = input_data
        self.out1 = out1
        self.out2 = out2
        self.out3 = out3

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        sample = {
            'input_data': self.input_data[index],
            'out1': self.out1[index], # position-position
            'out2': self.out2[index], # position-orientation
            'out3': self.out3[index] # orientation-orientation
        }
        return sample

