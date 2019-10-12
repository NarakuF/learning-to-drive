import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import Drive360Loader


class SomeDrivingModel(nn.Module):
    def __init__(self):
        super(SomeDrivingModel, self).__init__()
        final_concat_size = 0

        # Main CNN
        cnn = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(cnn.children())[:-1])
        self.intermediate = nn.Sequential(nn.Linear(
            cnn.fc.in_features, 128),
            nn.ReLU())
        final_concat_size += 128

        # Main LSTM
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=64,
                            num_layers=3,
                            batch_first=False)
        final_concat_size += 64

        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(final_concat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        module_outputs = []
        lstm_i = []
        # Loop through temporal sequence of
        # front facing camera images and pass
        # through the cnn.
        for k, v in data['cameraFront'].items():
            v = v.cuda()
            x = self.features(v)
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
        module_outputs.append(i_lstm[-1])

        # Concatenate current image CNN output
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)

        # Feed concatenated outputs into the
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))}
        return prediction


def train(train_loader, validation_loader, model, optimizer, criterion, epochs=1, use_validate=False):
    hist_mse_speed = []
    hist_mse_steer = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            prediction = model(data)

            # Optimizing for canSpeed and canSteering to optimize simultaneously.
            loss_speed = criterion(prediction['canSpeed'], target['canSpeed'].cuda())
            loss_steer = criterion(prediction['canSteering'], target['canSteering'].cuda())
            loss = loss_speed + loss_steer
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if batch_idx % 20 == 19:
                print('[epoch: %d, batch:  %d] loss: %.5f' %
                      (epoch + 1, batch_idx + 1, running_loss / 20.0))
                running_loss = 0.0

        if not use_validate:
            continue
        hist_mse_speed, hist_mse_steer = validate(validation_loader, model)
    return hist_mse_speed, hist_mse_steer


def validate(validation_loader, model):
    hist_mse_speed = []
    hist_mse_steer = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            prediction = model(data)
            diff_speed = prediction['canSpeed'].cuda() - target['canSpeed'].cuda()
            mse_speed = torch.mean(torch.mul(diff_speed, diff_speed))
            hist_mse_speed.append(mse_speed.cpu().detach().numpy())
            diff_steer = prediction['canSteering'].cuda() - target['canSteering'].cuda()
            mse_steer = torch.mean(torch.mul(diff_steer, diff_steer))
            hist_mse_steer.append(mse_steer.cpu().detach().numpy())
    return hist_mse_speed, hist_mse_steer


def test(test_loader, model, normalize, mean, std):
    results = {'canSteering': [],
               'canSpeed': []}
    i = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if len(data['cameraFront']) == 0:
                continue
            prediction = model(data)
            add_results(results, prediction, normalize, mean, std)
            i += 1
            if i % 100 == 0:
                print(i)
    return results


def add_results(results, output, normalize, mean, std):
    steering = np.squeeze(output['canSteering'].cpu().data.numpy())
    speed = np.squeeze(output['canSpeed'].cpu().data.numpy())
    if normalize:
        steering = (steering * std['canSteering']) + mean['canSteering']
        speed = (speed * std['canSpeed']) + mean['canSpeed']
    if np.isscalar(steering):
        steering = [steering]
    if np.isscalar(speed):
        speed = [speed]
    results['canSteering'].extend(steering)
    results['canSpeed'].extend(speed)


def extrapolate(target, ratio, results):
    test_full = pd.read_csv(target)['chapter']
    new_results = {'canSteering': [],
                   'canSpeed': []}
    chapter = -1
    idx = 0
    while idx < len(test_full):
        if chapter != test_full[idx]:
            chapter = test_full[idx]
            idx += 100
            print(chapter, idx)
        if idx >= len(test_full):
            break
        i = idx // ratio
        i = max(i - 1, 1)
        i = min(i, len(results['canSteering']) - 1)
        new_results['canSteering'].append(results['canSteering'][i])
        new_results['canSpeed'].append(results['canSpeed'][i])
        idx += 1
    return new_results


def main(load_model=True, save_result=False):
    # load the config.json file that specifies data
    # location parameters and other hyperparameters
    # required.
    config = json.load(open('./config.json'))
    path = config['path']

    torch.manual_seed(config['seed'])

    # create a train, validation and test data loader
    train_loader = Drive360Loader(config, 'train')
    validation_loader = Drive360Loader(config, 'validation')
    test_loader = Drive360Loader(config, 'test')

    # print the data (keys) available for use. See full
    # description of each data type in the documents.
    print('Loaded train loader with the following data available as a dict.')
    print(train_loader.drive360.dataframe.keys())

    # train the model
    if not load_model:
        model = SomeDrivingModel().cuda()
        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        train(train_loader, validation_loader, model, optimizer, criterion, 1, True)
        torch.save(model, path['model_path'])
    model = torch.load(path['model_path'])

    hist_speed, hist_steer = validate(validation_loader, model)
    plt.plot(hist_speed, label='canSpeed')
    plt.plot(hist_steer, label='canSteering')
    plt.show()
    print('MSE speed : %.5f' % np.average(hist_speed))
    print('MSE steer : %.5f' % np.average(hist_steer))

    if not save_result:
        return

    # test the model
    target = config['target']
    normalize_targets = target['normalize']
    target_mean = target['mean']
    target_std = target['std']
    results = test(test_loader, model, normalize_targets, target_mean, target_std)
    df = pd.DataFrame.from_dict(results)
    df.to_csv(path['downsampled_path'], index=False)

    ratio = {'full': 1,
             'sample1': 10,
             'sample2': 20,
             'sample3': 40}
    dataset = config['data_loader']['dataset']
    full_results = extrapolate(path['test_full_path'], ratio[dataset], results)

    # save to csv
    df = pd.DataFrame.from_dict(full_results)
    df.to_csv(path['output_path'], index=False)


if __name__ == '__main__':
    main()
