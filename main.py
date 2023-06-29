import torch
import torch.optim as optim
import pandas as pd
import random
from numpy import *
import Lib as L
from regulizer import *
from torch.optim import lr_scheduler

random.seed(2020)

def train_es():
    #
    dataset = r'/home/d/Fetal_Probelm/心脏数据集/Fetal-heart-classification/preprocess-data/train_test'
    writer = pd.ExcelWriter(r'/home/d/Fetal_Probelm/心脏数据集/Fetal-heart-classification/preprocess-data/acc' + str(i) + '.xlsx')
    model = L.load_model('resnet18', 5)
    model = model.to('cuda:0')
    train_data, test_data, train_data_size, test_data_size = L.load_data(dataset, batch_size=16)

    num_epochs = 50
    loss_func  = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    history = L.train_and_test(model, loss_func, optimizer, num_epochs, writer, train_data,
                                  test_data, train_data_size, test_data_size, dataset)
    L.plot_loss_and_acc(dataset, history)

def WD_train():
    for i in range(1, 2):
        dataset = r'/home/d/git_repo/LTR_weight_balancing/datasets'
        writer = pd.ExcelWriter(r'/home/d/git_repo/Image_classfication/acc' + str(i) + '.xlsx')
        num_epochs = 100
        model = L.load_model('densenet121', 6)
        model = model.to('cuda:0')
        weight_decay = 5e-3  # set weight decay value
        base_lr = 0.01
        optimizer = optim.SGD([{'params': model.parameters(), 'lr': base_lr}], lr=base_lr, momentum=0.9,
                              weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0)
        train_data, test_data, train_data_size, test_data_size = L.load_data(dataset, batch_size=64)
        loss_func = criterion = torch.nn.CrossEntropyLoss()
        history = L.train_and_test_WD(scheduler,model, loss_func, optimizer, num_epochs, writer, train_data,
                                      test_data, train_data_size, test_data_size, dataset)
        L.plot_loss_and_acc(dataset, history)

def L2_train():
    dataset = r'/home/d/git_repo/LTR_weight_balancing/datasets'
    writer = pd.ExcelWriter(r'/home/d/git_repo/Image_classfication/acc' + 'L2' + '.xlsx')
    model_ini = torch.load('/home/d/git_repo/LTR_weight_balancing/densenet121_WD/datasets_model_100.pt')
    model = copy.deepcopy(model_ini)
    # any changes made to "model" do not affect "model_ini"
    L2_norm = Normalizer(tau=1)
    L2_norm.apply_on(model)


def ft_maxnorm_wd_cb():
    dataset = r'/home/d/git_repo/LTR_weight_balancing/datasets'
    writer = pd.ExcelWriter(r'/home/d/git_repo/Image_classfication/acc' + str(3) + '.xlsx')
    num_epochs = 20
    model_ini = torch.load('/home/d/git_repo/Image_classification/12_14_classification_model_train/train_densenet121_model_100.pt')
    model_ini = model_ini.to('cuda:0')

    model = copy.deepcopy(model_ini)
    thresh = 0.1  # threshold value
    pgdFunc = MaxNorm_via_PGD(thresh=thresh)
    pgdFunc.setPerLayerThresh(model)  # set per-layer thresholds

    active_layers = [model.classifier.fc2.weight, model.classifier.fc2.bias]
    for param in model.parameters():  # freez all model paramters except the classifier layer
        param.requires_grad = False

    for param in active_layers:
        param.requires_grad = True

    base_lr = 0.01
    weight_decay = 0.1  # weight decay value
    optimizer = optim.SGD([{'params': model.parameters(), 'lr': base_lr}], lr=base_lr, momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0)
    # This scheduler is used to adjust the learning rate of the optimizer during training.

    train_data, test_data, train_data_size, test_data_size = L.load_data(dataset, batch_size=64)
    loss_func = torch.nn.CrossEntropyLoss()

    history = L.train_and_test_WD_CB(scheduler, model, loss_func, optimizer, num_epochs, writer, train_data,
                                  test_data, train_data_size, test_data_size, dataset)

    L.plot_loss_and_acc(dataset, history)




