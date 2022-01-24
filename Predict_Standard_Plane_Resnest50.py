#from Load_Inhosp_No_divided_v1 import Data

import pretrainedmodels
from torch import optim, nn
#from Load_Data_After_Choose_0310 import Data
from Load_Data_After_Choose_200_0428 import Data

#from Load_Inhosp_No_divided_v1 import Data
import pretrainedmodels
from torch import optim, nn
#from Load_data_10_14 import Data

from resnest.torch import resnest50
# from    resnet import ResNet18
from torchvision.models import resnet18
from torchvision.models import resnet50
import torch
from torch import optim, nn
import visdom
import torchvision
from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score
from resnest.torch import resnest50
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision import models

import numpy as np
from sklearn import metrics

import time
import numpy as np

import matplotlib.pyplot as plt
#f#rom efficientnet_pytorch import EfficientNet
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from util import Flatten
from resnest.torch import resnest50
# from    resnet import ResNet18
from torchvision.models import resnet18
from torchvision.models import resnet50
import torch
from torch import optim, nn
import visdom
import torchvision
from resnest.torch import resnest50
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision import models

import numpy as np
from sklearn import metrics

import time
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from util import Flatten
from torch.optim.lr_scheduler import *


device = torch.device('cuda')
torch.manual_seed(1234)




batchsz =32

epochs =200

print("*******Start training******")
print("共{}epoch".format(epochs))


train_db = Data('./Plane', 224, mode='train')
train_loader = DataLoader(train_db, batch_size=batchsz,  shuffle=True,
                          num_workers=0)


val_db = Data('./Plane', 224, mode='val')
# val_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 327)
val_loader = DataLoader(val_db, batch_size=batchsz,  shuffle=True,
                         num_workers=0)


test_db = Data('./Plane', 224, mode='test')
test_loader = DataLoader(test_db, batch_size=batchsz,shuffle=True,  num_workers=0)


viz = visdom.Visdom()



def confusion_matrixes(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    y_true = []
    y_scores = []
    for Inhosp_No, EXAM_NO, x, y in loader:
        y_true.extend(y)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_score = model(x)
            # print("y_score",y_score)

            #             pred = logits.argmax(dim=1)
            y_scores.append(y_score)

    y_true = torch.tensor([i.item() for i in y_true])

    y_true = to_categorical(y_true, 2)
    # print("y_true",y_true)
    y_scores = torch.cat([i for i in y_scores], 0)
    # print("y_scores",y_scores)

    # _scores= torch.tensor([i.item() for i in y_scores])
    #     print("y_pred",y_pred)
    #     print("y_true",y_true)
    y_scores = y_scores.cpu().numpy()
    # y_true=y_true.cpu().numpy()
    auc = roc_auc_score(y_true, y_scores)

    return auc


def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for Inhosp_No, EXAM_NO, x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            y_pred = logits[:, 1]
            print(y_pred)
            y_pred = y_pred > 0.4
            # y_pred=y_pred.cpu().detach().numpy()
            # y=y.cpu().detach().numpy()
            # y_pred=y_pred.astype(int)
            y_pred = y_pred.int()

        #             pred=logits>0.4
        #             pred=pred.astype(int)
        #             pred = logits.argmax(dim=1)

        correct += torch.eq(y_pred, y).sum().float().item()

    return correct / total


def main():
    trained_model = resnest50(pretrained=True)
    # #     trained_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                          Flatten(),  # [b, 512, 1, 1] => [b, 512]
                          # nn.Linear(2048, 512),
                          # nn.Linear(512, 2),
                          nn.BatchNorm1d(2048, affine=True),
                          nn.Dropout(0.29),
                          nn.Linear(2048, 2),
                          nn.BatchNorm1d(2, affine=True),
                          # nn.Linear(248, 2),
                          nn.Softmax(dim=1)
                          ).to(device)

    #optimizer = optim.Adam(model.parameters(), lr=0.008, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.04,
                        #   amsgrad=False)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001,
                           amsgrad=False)
    #scheduler = CosineAnnealingLR(optimizer, T_max=5)
    scheduler= ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=False, threshold=0.0001,
                       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
   #
    # lr_list = []
    # for epoch in range(epochs):
    #     if epoch % 10 == 0:
    #         for p in optimizer.param_groups:
    #             p['lr'] *= 0.9
    #     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    #lr_list = []
    # for epoch in range(epochs):
    #     if epoch % 20 == 0:
    #         for p in optimizer.param_groups:
    #             p['lr'] *= 0.9
    #     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    #
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    # criteon = nn.BCELoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    #viz.line([0], [-1], win='loss', opts=dict(title='loss'))

    viz.line([0], [-1], win='val_loss', opts=dict(title='val_loss'))

    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    start = time.time()
    for epoch in range(epochs):
        # scheduler.step()
        train_loss = 0.0

        for step, (Inhosp_No, EXAM_NO, x, y) in enumerate(train_loader):
            #             y_trains.extend(y)

            print("********Enter {} 的{} step ***********".format(epoch, step))
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            #             train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


            #lr = scheduler.get_lr()
           # print(epoch, scheduler.get_lr()[0])
            #lr_list.append(scheduler.get_lr()[0])

            #viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1
            train_loss += loss.item() * x.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_acc = evalute(model, train_loader)
        train_accs.append(train_acc)

        if epoch % 1 == 0:
            val_loss = 0.0
            for Inhosp_No, EXAM_NO, x, y in val_loader:
                # x, y = x.to(device), y.unsqueeze(1).to(device)
                x, y = x.to(device), y.to(device)
                #                 ys.extend(y)
                with torch.no_grad():
                    logits_val = model(x)
                    loss_val = criteon(logits_val, y)
                    val_loss += loss_val.item() * x.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            viz.line([val_loss], [epoch], win='val_loss', update='append')
            scheduler.step(val_loss)
            # print("val_loss",val_loss)
            val_losses.append(val_loss)
            val_acc = evalute(model, val_loader)
            viz.line([val_acc], [epoch], win='val_acc', update='append')
            val_accs.append(val_acc)

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                #torch.save(model.state_dict(), 'Resnest50_0429.mdl') #2021-0601 结果有效 0.0001
                torch.save(model.state_dict(), 'Resnest50_0601.mdl')  #

    print("val_losses", val_losses)
    print("train_losses", train_losses)
    print('best acc:', best_acc, 'best epoch:', best_epoch)
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = train_accs
    import pandas as pd
    train_accs_y1=pd.DataFrame(train_accs)
    #train_accs_y1.to_csv("train_accs_y1_2021_0426.csv")
    #train_accs_y1.to_csv("train_accs_y1_Resnest50_2021_0429.csv") #有效

    train_accs_y1.to_csv("train_accs_y1_Resnest50_2021_0601.csv")
    # y1 = Accuracy_list
    y2 = val_accs
    val_accs_y2 = pd.DataFrame(val_accs)
   # val_accs_y2.to_csv("val_accs_y2_2021_0426.csv")
   # val_accs_y2.to_csv("val_accs_y2_Resnest50_2021_0429.csv")#有效
    val_accs_y2.to_csv("val_accs_y2_Resnest50_2021_0601.csv")


    y3 = train_losses
    train_losses_y3 = pd.DataFrame(train_losses)
    #train_losses_y3.to_csv("train_losses_y3_2021_0426.csv")
    #train_losses_y3.to_csv("train_losses_y3_Resnest50_2021_0429.csv")#有效
    train_losses_y3.to_csv("train_losses_y3_Resnest50_2021_0601.csv")

    y4 = val_losses
    val_losses_y4 = pd.DataFrame(val_losses)
    #val_losses_y4.to_csv("val_losses_y4_2021_0426.csv")
    #val_losses_y4.to_csv("val_losses_y4_Resnest50_2021_0429.csv")#有效
    val_losses_y4.to_csv("val_losses_y4_Resnest50_2021_0601.csv")
    # y2 = Loss_list
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1, color='lime')
    plt.plot(x1, y2, color='r')
    # plt.title('Cross_validation curve')
    plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    plt.subplot(2, 2, 3)

    plt.plot(x2, y3, color='lime')
    plt.plot(x2, y4, color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entrotopy Loss')
    # plt.legend(('No mask', 'Masked if > 0.5', 'Masked if < -0.5'), loc='upper right')
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(range(epochs), lr_list, color='r')
    #
    # plt.ylabel('lr')
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(lr_list, val_losses, color='r')
    # plt.ylabel('val_losses')

    # plt.savefig("Resnest50_0426_accuracy_loss.pdf")
    # plt.savefig("Resnest50_0426_accuracy_loss.svg")

    # plt.savefig("Resnest50_0429_accuracy_loss.pdf")
    # plt.savefig("Resnest50_0429_accuracy_loss.svg")
    #plt.savefig("Resnest50_0531_accuracy_loss.svg")  #图有效
    #plt.savefig("Resnest50_0601_accuracy_loss.svg")#图有效
    plt.savefig("Resnest50_0602_accuracy_loss.svg")
    plt.show()

   # model.load_state_dict(torch.load('Resnest50_0426.mdl'))
    model.load_state_dict(torch.load('Resnest50_0601.mdl'))

    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)
    torch.cuda.synchronize()
    end = time.time()
    GPU_time = end - start
    print("GPU_TIME=", GPU_time)
    num_params = sum(param.numel() for param in model.parameters())
    print('num_params', num_params)

    AUC_train = confusion_matrixes(model, train_loader)
    print("confusion_train", AUC_train)
    AUC_val = confusion_matrixes(model, val_loader)
    print("confusion_val", AUC_val)
    AUC_test = confusion_matrixes(model, test_loader)
    print("confusion_test", AUC_test)


if __name__ == '__main__':
    main()
