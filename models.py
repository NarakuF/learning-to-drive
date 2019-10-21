from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda")


class SomeDrivingModel(nn.Module):
    def __init__(self):
        super(SomeDrivingModel, self).__init__()
        final_concat_size = 0

        # Main CNN
        cnn = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        self.intermediate = nn.Sequential(
            nn.Linear(cnn.fc.in_features, 128),
            nn.ReLU(inplace=True)
        )
        final_concat_size += 128

        # Main LSTM
        self.lstm = nn.LSTM(input_size=128,
                           hidden_size=64,
                           num_layers=3,
                           batch_first=False,
                           dropout=0.1)
        final_concat_size += 64

        final_concat_size = 64
        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        # self.attn = nn.Linear(64, 1)

    def forward(self, data):
        module_outputs = []
        lstm_i = []
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn.
        for k, v in data['cameraFront'].items():
            x = v.cuda()
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.intermediate(x)
            lstm_i.append(x)
            # feed the current front facing camera
            # output directly into the 
            # regression networks.
            if k == 0:
                module_outputs.append(x)

        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm(torch.stack(lstm_i))

        x_cat = i_lstm[-1]
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}

        return prediction
        '''

        # Feed temporal outputs of CNN into LSTM
        i_lstm, _ = self.lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])

        # Concatenate current image CNN output
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction'''
