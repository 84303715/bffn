import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import random_split

import os
import numpy as np
from dataset import Dataset

from res_18 import FasterRCNNResnet18
from res_101 import FasterRCNNResnet101


EPOCHS = 400
BATCH_SIZE = 32

# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签
def calculate_acuracy_mode_one(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item(), recall.item()

def main():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")   

    model = FasterRCNNResnet18()

    model.to(device)

    data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    datasets = Dataset(transform=data_transforms)

    train_size = int(datasets.__len__() * 0.7)
    test_size = int(datasets.__len__() - train_size)
    train_datasets, test_datasets = random_split(dataset=datasets, 
                        lengths=[train_size, test_size]
                        )
    print("Train dataset size: {0} Test dataset size: {1}".format(train_size, test_size))
    training_loader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

    loss_fn = torch.nn.BCELoss()
    best_precision = 0.

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.
        running_precision = 0.
        running_recall = 0.
        iter_cnt = 0

        for i, data in enumerate(training_loader):
            iter_cnt += 1
            idxs, imgs, orig_box_lsts, labels = data

            imgs = imgs.cuda()
            orig_box_lsts = orig_box_lsts.cuda()
            labels = labels.cuda()

            outputs = model(imgs, orig_box_lsts)

            loss = loss_fn(torch.sigmoid(outputs.float()), labels.float())
    
            precision, recall = calculate_acuracy_mode_one(torch.sigmoid(outputs), labels)
            running_precision += precision
            running_recall += recall

            # Backropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_datasets)
        try:
            train_f1 = 2 * ((running_precision / iter_cnt) * (running_recall / iter_cnt)) / ((running_precision / iter_cnt) + (running_recall / iter_cnt))
        except ZeroDivisionError as e:
            train_f1 = 0
        print('[Epoch %d] Training loss: %.4f. Train precision: %.4f. Train recall: %.4f. Train f1: %.4f' %
                    (epoch + 1, train_loss, running_precision / iter_cnt, running_recall / iter_cnt, train_f1))

        scheduler.step()

    
        with torch.no_grad():

            test_loss = 0.
            running_precision = 0.
            running_recall = 0.
            iter_cnt = 0

            for i, data in enumerate(test_loader):
                iter_cnt += 1
                idxs, imgs, orig_box_lsts, labels = data

                imgs = imgs.cuda()
                orig_box_lsts = orig_box_lsts.cuda()
                labels = labels.cuda()

                outputs = model(imgs, orig_box_lsts)

                loss = loss_fn(torch.sigmoid(outputs.float()), labels.float())
        
                precision, recall = calculate_acuracy_mode_one(torch.sigmoid(outputs), labels)
                running_precision += precision
                running_recall += recall


                test_loss += loss.item() * imgs.size(0)
            test_loss = test_loss / len(test_datasets)
            try:
                test_f1 = 2 * ((running_precision / iter_cnt) * (running_recall / iter_cnt)) / ((running_precision / iter_cnt) + (running_recall / iter_cnt))
            except ZeroDivisionError as e:
                test_f1 = 0

            print('[Epoch %d] Test loss: %.4f. Test precision: %.4f. Test recall: %.4f. Test f1: %.4f. ' %
                        (epoch + 1, test_loss, running_precision / iter_cnt, running_recall / iter_cnt, test_f1))
            if (running_precision / iter_cnt) > best_precision and (running_precision / iter_cnt) > 0.68:

                best_precision = running_precision / iter_cnt
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                            os.path.join(r"C:\Users\79288\Desktop\Fair_LRN-latest\AU\checkpoints", "resnet-34-epoch_" + str(epoch+1) + "_precision_" + str(best_precision) + ".pth"))
                print('Model saved.')


if __name__ == "__main__":
    main()

