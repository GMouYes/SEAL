import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util import labelText
from transformers import AutoTokenizer 


class myDataset(Dataset):
    """ dataset reader
    """

    def __init__(self, args, dataType):
        super(myDataset, self).__init__()

        self.x = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["hcPath"] + '.npy'), dtype=torch.float)
        self.y = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["yPath"] + '.npy'), dtype=torch.float)
        self.weight = torch.tensor(np.load(args["dataPath"] + dataType + '/' + args["weightPath"] + '.npy'), dtype=torch.float)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.weight[index]


def loadData(args, dataType):
    data = myDataset(args, dataType)
    # print("{} size: {}".format(dataType, len(data.y)))

    shuffle = (dataType == "train")
    if shuffle:
        loader = DataLoader(data, batch_size=args["batch_size"], shuffle=shuffle, num_workers=args["num_workers"],
                            pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(data, batch_size=args["batch_size"], shuffle=shuffle, num_workers=args["num_workers"],
                            pin_memory=True)
    return loader

def getText():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer(labelText, padding=True, return_tensors="pt")