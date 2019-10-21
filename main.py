import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Drive360Loader
from models import SomeDrivingModel


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
            if batch_idx % 50 == 49:
                print('[epoch: %d, batch:  %d] loss: %.5f' %
                      (epoch + 1, batch_idx + 1, running_loss / 50.0))
                running_loss = 0.0

        if not use_validate:
            continue
        val_mse_speed, val_mse_steer = validate(validation_loader, model)
        hist_mse_speed.append(np.mean(val_mse_speed))
        hist_mse_steer.append(np.mean(val_mse_steer))
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


def experiment(test_loader, model, normalize, mean, std):
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


def recover(results, test, max_temporal_history=8):
    pre_chapter = -1
    new_results = {"canSteering": [],
                   "canSpeed": []}
    result_idx = 0
    i = 0
    while i < test.shape[0]:
        # print(i, len(new_result["canSteering"]), result_idx)
        cur_chapter = list(test.chapter)[i]
        if pre_chapter != cur_chapter:
            new_results["canSteering"] += [results["canSteering"][result_idx]] * max_temporal_history
            new_results["canSpeed"] += [results["canSpeed"][result_idx]] * max_temporal_history
            pre_chapter = cur_chapter
            i += max_temporal_history
        else:
            new_results["canSteering"].append(results["canSteering"][result_idx])
            new_results["canSpeed"].append(results["canSpeed"][result_idx])
            result_idx += 1
            i += 1
    return new_results


def extrapolate(results, target, ratio):
    test_full = pd.read_csv(target)['chapter']
    new_results = {'canSteering': [],
                   'canSpeed': []}
    chapter = -1
    # index of full test
    idx = 0
    while idx < len(test_full):
        if chapter != test_full[idx]:
            chapter = test_full[idx]
            idx += 100
            print(chapter, idx)
        if idx >= len(test_full):
            break
        # index of sample test
        i = (idx + 1) // ratio
        offset = (idx + 1) % ratio
        if i == 0 or i > len(results['canSteering']) - 1:
            i = min(i, len(results['canSteering']) - 1)
            new_steering = results['canSteering'][i]
            new_speed = results['canSpeed'][i]
        else:
            new_steering = results['canSteering'][i]
            new_speed = results['canSpeed'][i]
            prev_steering = results['canSteering'][i - 1]
            new_steering = prev_steering + (offset / ratio) * (new_steering - prev_steering)
            prev_speed = results['canSpeed'][i - 1]
            new_speed = prev_speed + (offset / ratio) * (new_speed - prev_speed)
        new_results['canSteering'].append(new_steering)
        new_results['canSpeed'].append(new_speed)
        idx += 1
    return new_results


def main(model_name='model.pth', load_model=False, save_result=False):
    # load the config.json file that specifies data
    # location parameters and other hyperparameters
    # required.
    config = json.load(open('./config.json'))
    path = config['path']

    torch.manual_seed(config['seed'])

    # create a train, validation and test data loader
    dataset = config['data_loader']['dataset']
    csv_dir = path['csv_dir']
    test_csv = pd.read_csv(f'{csv_dir}test_{dataset}.csv')
    full_test_csv = csv_dir + path['test_full_path']
    train_loader = Drive360Loader(config, 'train')
    val_loader = Drive360Loader(config, 'val')
    test_loader = Drive360Loader(config, 'test')

    # print the data (keys) available for use. See full
    # description of each data type in the documents.
    print('Loaded train loader with the following data available as a dict.')
    print(train_loader.drive360.dataframe.keys())

    # train the model
    hist_speed = []
    hist_steer = []
    if not load_model:
        model = SomeDrivingModel().cuda()
        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        hist_speed, hist_steer = train(train_loader, val_loader, model, optimizer, criterion,
                                       epochs=5, use_validate=True)
        torch.save(model, path['model_dir'] + model_name)
    model = torch.load(path['model_dir'] + model_name)

    # hist_speed, hist_steer = validate(val_loader, model)
    plt.plot(hist_speed, label='canSpeed')
    plt.plot(hist_steer, label='canSteering')
    plt.show()
    print('MSE speed : %.5f' % np.sum(hist_speed))
    print('MSE steer : %.5f' % np.sum(hist_steer))

    # test the model
    target = config['target']
    normalize_targets = target['normalize']
    target_mean = target['mean']
    target_std = target['std']
    results = experiment(test_loader, model, normalize_targets, target_mean, target_std)

    # save the model
    if not save_result:
        return
    df = pd.DataFrame.from_dict(results)
    downsampled = f'{csv_dir}results_{dataset}.csv'
    df.to_csv(downsampled, index=False)

    print(len(results['canSpeed']))
    max_temporal_history = config['data_loader']['historic']['number'] * \
                           config['data_loader']['historic']['frequency']
    recovered_results = recover(results, test_csv, max_temporal_history)
    print(len(recovered_results['canSpeed']))
    ratio = {'full': 1, 'sample1': 10, 'sample2': 20, 'sample3': 40}
    full_results = extrapolate(recovered_results, full_test_csv, ratio[dataset])

    df = pd.DataFrame.from_dict(full_results)
    df.to_csv(csv_dir + path['output_path'], index=False)


if __name__ == '__main__':
    main('sample3_new.pth', False, save_result=False)
