import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import torch.utils.data as data_utils


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        # start with 1 hidden layer of 64 units
        self.fc1 = nn.Linear(self.input_size, 64)
        self.layer_out = nn.Linear(64, 1)

    def forward(self, x):
        # TODO: explore dropout
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.layer_out(x))


class MyNet():
    """Wrapper class for neural network

    """
    def __init__(self, input_size: int, num_epochs: int, batch_size: int,
                 X: pd.DataFrame, y: pd.Series, save_path: str):
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.save_path = save_path

    # Code taken from https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor
    # Anh-Thi DINH
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        return device

    # Code taken from https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor
    # Gaurav Shrivastava
    def df_to_trainloader(self) -> data_utils.DataLoader:
        # see https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
        train = data_utils.TensorDataset(torch.Tensor(self.X.to_numpy()), torch.Tensor(self.y.to_numpy()))
        train_data = data_utils.DataLoader(train, batch_size = self.batch_size, shuffle=True)
        return train_data


    # see PyTorch tutorial
    def train_nn(self) -> None:
        net = Net(self.input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.09)

        device = self.get_device()
        data_loader = self.df_to_trainloader()

        for epoch in range(self.num_epochs):

            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                if i % 2000 == 1999: # print every 2000 mini-batches
                    print(f"[{epoch +1} {i+1}] loss: {running_loss / 2000}")
                    running_loss = 0.0

        return nn