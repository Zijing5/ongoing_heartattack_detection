import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from PIL import Image
import os
import cv2
import numpy as np
import torch.nn.functional as F
import pandas as pd
import random
import shutil
import xlrd
from numpy import *
from torch.utils.data import Dataset

def load_model(model_name, i):
    """（1）如果使用这个函数调用，那么必须要是下面的这些网络才可以
    densenet121, resnet18, resnet101, resnet152, googlenet, inception-v3;
    （2）另外，多设置了一个i参数，如果想要改不同的分类，比如分类数为5，调用时实参设置为5即可
    （3）使用范例：model=load_model('resnet18',3)
    （4）因为最后的分类器也在这里修改，如果想加丢弃层，就在这里改"""
    # please attention that the "model_name" must in the list below:
    if model_name == 'densenet':

        densenet121 = models.densenet121(pretrained=True)
        for param in densenet121.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('drop',nn.Dropout(0.5)),
            ('fc1', nn.Linear(1024, 500)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(500, i)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        densenet121.classifier = classifier
        ''' features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)——>self.classifier = nn.Linear(num_features, num_classes)'''
        model = densenet121.to('cuda:0')
        return model

    elif model_name == 'resnet101':
        resnet101 = models.resnet101(pretrained=True)
        # for param in resnet101.parameters():
        #     param.requires_grad = False

        fc_inputs = resnet101.fc.in_features
        resnet101.fc = nn.Sequential(
            nn.Linear(2048, i)
        )
        '''last layer in resnet source code: self.fc = nn.Linear(512 * block.expansion, num_classes)'''
        model = resnet101.to('cuda:0')
        return model

    elif model_name == 'googlenet':
        googlenet = models.googlenet(pretrained=True)
        # for param in googlenet.parameters():
        #     param.requires_grad = False
        fc_inputs = googlenet.fc.in_features
        googlenet.fc = nn.Sequential(
            nn.Linear(1024, i)
        )
        model = googlenet.to('cuda:0')
        return model

    elif model_name == 'resnet152':
        resnet152 = models.resnet152(pretrained=True)
        for param in resnet152.parameters():
            param.requires_grad = False
        fc_inputs = resnet152.fc.in_features
        resnet152.fc = nn.Sequential(
            nn.Linear(2048, i)
        )
        model = resnet152.to('cuda:0')
        return model

    elif model_name == 'inception_v3':
        inception_v3 = models.inception_v3(pretrained=True)
        for param in inception_v3.parameters():
            param.requires_grad = False
        inception_v3.aux_logits = False
        fc_inputs = inception_v3.fc.in_features

        inception_v3.fc = nn.Sequential(
            nn.Linear(2048, i)
        )

        model = inception_v3.to('cuda:0')
        return model

    elif model_name == 'resnet':
        resnet18 = models.resnet18(pretrained=True)
        # for param in resnet18.parameters():
        #     param.requires_grad = False

        # fc_inputs = resnet18.fc.in_features
        resnet18.fc = nn.Sequential(
            nn.Linear(512, i)
        )
        model = resnet18.to('cuda:0')
        return model

    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        model.heads = nn.Linear(768,6)
        model = model.to('cuda:0')
        return model

    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(1280, 6),
        )
        model = model.to('cuda:0')
        return model
    else:
        print('no model you need here')


def load_data(dataset, batch_size):
    """(1)dataset是图片的路径，batch_size是训练的批量大小，
       (2)返回值是train_data,test_data,train_data_size,test_data_size
       (3)调用示例：train_data,test_data=load_data(dataset,batch_size)
       （4）设置这么多返回值的原因是emmm忘了，反正是发现后面用到才会重新加的"""
    # 对数据进行预处理
    image_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(size=256,scale=(0.8,1.0)),
            transforms.CenterCrop(size=224),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([
            # transforms.RandomResizedCrop(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])  # 字典：键值对的用法
    }

    # 加载数据
    train_directory = os.path.join(dataset, 'train')
    test_directory = os.path.join(dataset, 'test')

    batch_size = batch_size
    feature_extract = True
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test']),
    }
    train_data_size = len(data['train'])
    test_data_size = len(data['test'])

    # train_data = DataLoader(data['train'], sampler=ImbalancedDatasetSampler(data['train']), batch_size=batch_size)
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

    return train_data, test_data, train_data_size, test_data_size


def default_loader(path):
    try:
        # img = cv2.imread(path)  # b,h,w,c
        # return img
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


class Data(Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines] # should be ['img1path','img2path','img3path'...]
            self.img_label = [line.strip('\n').split(' ')[-1] for line in lines] # should be ['label1path','label2path','label3path'...]

        self.data_transforms = data_transforms
        self.dataset = dataset #可以删除
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item] # is it actually index?
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label


def load_data_split(dataset, batch_size):
    # 对数据进行预处理
    image_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(size=256,scale=(0.8,1.0)),
            transforms.CenterCrop(size=224),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.RandomResizedCrop(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])  # 字典：键值对的用法
    }
    batch_size = batch_size
    # 加载数据
    train_image_path = os.path.join(dataset, 'train')
    train_txt = os.path.join(dataset, 'train.txt')

    test_image_path = os.path.join(dataset, 'test')
    test_txt = os.path.join(dataset, 'test.txt')

    train_data = DataLoader(Data(train_image_path, train_txt, data_transforms=image_transforms['train']),
                            batch_size=batch_size, shuffle=True)
    test_data = DataLoader(Data(test_image_path, test_txt, data_transforms=image_transforms['test']),
                           batch_size=batch_size, shuffle=True)

    return train_data, test_data


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=30, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def train_and_test(model, loss_function, optimizer, epochs, writer,
                   train_data, test_data, train_data_size,test_data_size, dataset,
                   saved_model):
    """（1）参数说明：model是要跑的模型，loss,optimizer,epochs，writer是acc要存放的excel表格，data_record是存放acc的字典
       （2）函数的返回值只有history,训练过的model已经自动保存，data_record也在代码运行过程中写进了excel表格中
       （3）调用示例：history=train_and_test(model,loss_func,optimizer,num_epochs,writer,data_record)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    data = {'Epoch': [],
            'Train_acc': [],
            'Test_acc': [],
            'Train_loss':[],
            'Test_loss':[]}
    # early_stopping = early_stopping
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc_test = 0.0
    best_acc_train = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            #            exp_lr_scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                test_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
        data['Epoch'].append(best_epoch)
        data['Train_acc'].append(avg_train_acc)
        data['Test_acc'].append(avg_test_acc)
        data['Train_loss'].append(avg_train_loss)
        data['Test_loss'].append(avg_test_loss)

        if best_acc_test < avg_test_acc and best_acc_train < avg_train_acc:
            best_acc_test = avg_test_acc
            best_acc_train = avg_train_acc
            best_epoch = epoch + 1
            save_path = os.path.join(saved_model,str(epoch + 1) + '.pt')
            torch.save(model, save_path)
        #             torch.save(model.state_dict(),dataset+'_'+str(epoch+1)+'only_best_net_parameters.pt')
        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc_test, best_epoch))

        if epoch == (epochs - 1):
            save_path = os.path.join(saved_model,str(epoch + 1) + '.pt')
            torch.save(model, save_path)

    df = pd.DataFrame(data)
    df.to_excel(writer)
    writer.save()
    return history


def train_and_test_es(model, loss_function, optimizer, epochs, early_stopping, writer,
                      train_data, test_data,train_data_size, test_data_size, dataset,
                      saved_model):
    """（1）参数说明：model是要跑的模型，loss,optimizer,epochs，writer是acc要存放的excel表格，data_record是存放acc的字典
       （2）函数的返回值只有history,训练过的model已经自动保存，data_record也在代码运行过程中写进了excel表格中
       （3）调用示例：history=train_and_test(model,loss_func,optimizer,num_epochs,writer,data_record)
       （4）后来又加了早停，所以这是一个有早停的版本"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    data = {'Epoch': [],
            'Train_acc': [],
            'Test_acc': [],
            'Train_loss':[],
            'Test_loss': [],
            }
    early_stopping = early_stopping
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc_test = 0.0
    best_acc_train = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            # exp_lr_scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                test_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])

        # 早停，如果停止在某一轮，则保存模型，保存acc
        early_stopping(avg_test_loss, model)
        if early_stopping.early_stop:
            # 修改了最后保存的模型名称，再不用修改模型名称了
            torch.save(model, dataset + '_model.pt')
            data['Epoch'].append(epoch + 1)
            data['Train_acc'].append(avg_train_acc)
            data['Test_acc'].append(avg_test_acc)
            print("Early stopping")
            break

        if best_acc_test < avg_test_acc and best_acc_train < avg_train_acc:
            best_acc_test = avg_test_acc
            best_acc_train = avg_train_acc
            best_epoch = epoch + 1

            data['Epoch'].append(best_epoch)
            data['Train_acc'].append(best_acc_train)
            data['Test_acc'].append(best_acc_test)

            torch.save(model, dataset + '_model_' + str(epoch + 1) + '.pt')
        #             torch.save(model.state_dict(),dataset+'_'+str(epoch+1)+'only_best_net_parameters.pt')
        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc_test, best_epoch))

        if epoch == (epochs - 1):
            torch.save(model, dataset + '_model_' + str(epoch + 1) + '.pt')
            #             torch.save(model.state_dict(),dataset+str(epoch+1)+'only_best_net_parameters.pt')

            # add code here
            data['Epoch'].append(epoch + 1)
            data['Train_acc'].append(avg_train_acc)
            data['Test_acc'].append(avg_test_acc)

    torch.save(history, dataset + '_history.pt')

    df = pd.DataFrame(data)
    df.to_excel(writer)
    writer.save()
    return history


def plot_loss_and_acc(dataset, history):
    """
   （1）dataset为输入图片要保存的路径名称，
   （2）history是列表形式的文件，保存了epochs*4(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc)的数据
       history中的数据由train函数返回。
   （3）函数无返回值
    (4)调用示例：plot_loss_and_acc(dataset,history)"""

    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val loss'])
    plt.xlabel('epoch number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_loss_curve.png')
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_accuracy_curve.png')
    plt.show()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def train_and_test_WD(scheduler, model, loss_function, optimizer, epochs, writer,
                      train_data, test_data,
                      train_data_size,test_data_size, dataset,
                      saved_model):
    """（1）参数说明：model是要跑的模型，loss,optimizer,epochs，writer是acc要存放的excel表格，data_record是存放acc的字典
       （2）函数的返回值只有history,训练过的model已经自动保存，data_record也在代码运行过程中写进了excel表格中
       （3）调用示例：history=train_and_test(model,loss_func,optimizer,num_epochs,writer,data_record)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    data = {'Epoch': [],
            'Train_acc': [],
            'Test_acc': [],
            'Train_loss': [],
            'Test_loss': []
            }
    # early_stopping = early_stopping
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc_test = 0.0
    best_acc_train = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            # weight-decay here
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                test_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
        data['Epoch'].append(best_epoch)
        data['Train_acc'].append(avg_train_acc)
        data['Test_acc'].append(avg_test_acc)
        data['Train_loss'].append(avg_train_loss)
        data['Test_loss'].append(avg_test_loss)

        if best_acc_test < avg_test_acc and best_acc_train < avg_train_acc:
            best_acc_test = avg_test_acc
            best_acc_train = avg_train_acc
            best_epoch = epoch + 1
            save_path = os.path.join(saved_model,str(epoch + 1) + '.pt')
            torch.save(model, save_path)

        #             torch.save(model.state_dict(),dataset+'_'+str(epoch+1)+'only_best_net_parameters.pt')
        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc_test, best_epoch))

        if epoch == (epochs - 1):
            save_path = os.path.join(saved_model,str(epoch + 1) + '.pt')
            torch.save(model, save_path)


    df = pd.DataFrame(data)
    df.to_excel(writer)
    writer.save()
    return history


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    labels_one_hot = labels_one_hot.to(device)

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1).to(device)
    weights = weights * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    # if loss_type == "focal":
    #     cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    if loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weights=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss



def CB_lossFunc(logits, labelList):  # defince CB loss function
    img_num_per_cls = [524, 132, 234, 121, 110, 324]
    nClasses = 6
    device = torch.device('cuda:0')
    return CB_loss(labelList, logits, img_num_per_cls, nClasses, "softmax", 0.9999, 2.0, device)


def train_and_test_WD_CB(scheduler, model, loss_function, optimizer, epochs, writer,
                         train_data, test_data,train_data_size,test_data_size,
                         dataset,saved_data_path):
    """（1）参数说明：model是要跑的模型，loss,optimizer,epochs，writer是acc要存放的excel表格，data_record是存放acc的字典
       （2）函数的返回值只有history,训练过的model已经自动保存，data_record也在代码运行过程中写进了excel表格中
       （3）调用示例：history=train_and_test(model,loss_func,optimizer,num_epochs,writer,data_record)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    data = {'Epoch': [],
            'Train_acc': [],
            'Test_acc': [],
            'Train_loss': [],
            'Test_loss': []
            }
    # early_stopping = early_stopping
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc_test = 0.0
    best_acc_train = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        """     imageList, labelList = sample
                imageList = imageList.to(device)
                labelList = labelList.type(torch.long).view(-1).to(device)

                # zero the parameter gradients
                optimizerW.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    logits = model(imageList)
                    error = lossFunc(logits, labelList)
                    softmaxScores = logits.softmax(dim=1)"""

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = CB_lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()
            # using Weight_decay
            scheduler.step() # for lr decay

            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                test_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        data['Epoch'].append(best_epoch)
        data['Train_acc'].append(avg_train_acc)
        data['Test_acc'].append(avg_test_acc)
        data['Train_loss'].append(avg_train_loss)
        data['Test_loss'].append(avg_test_loss)

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])

        if best_acc_test < avg_test_acc and best_acc_train < avg_train_acc:
            best_acc_test = avg_test_acc
            best_acc_train = avg_train_acc
            best_epoch = epoch + 1

            save_path = os.path.join(saved_data_path,str(epoch + 1) + '.pt')
            torch.save(model, save_path)
        #             torch.save(model.state_dict(),dataset+'_'+str(epoch+1)+'only_best_net_parameters.pt')
        epoch_end = time.time()
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc_test, best_epoch))

        if epoch == (epochs - 1):
            save_path = os.path.join(saved_data_path,str(epoch + 1) + '.pt')
            torch.save(model, save_path)

    df = pd.DataFrame(data)
    df.to_excel(writer)
    writer.save()
    return history

