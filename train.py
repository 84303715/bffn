import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import argparse
from tqdm import tqdm
import numpy as np
from dataset import Dataset

import utils
from faster_rcnn_resnet50 import FasterRCNNResnet50




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="./dataset/raf-db", help='dataset path.')
    parser.add_argument('--bs', type=int, default=16, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.007, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd (default: 0.9)')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight_decay for sgd (default: 0.01)')
    parser.add_argument('--step_size', default=10, type=int, help='step_size for lr_scheduler (default: 20)')
    parser.add_argument('--gamma', default=0.8 , type=float, help='gamma for lr_scheduler (default: 0.8)')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--lamb', type=float, default=0.6, help='Weight for bmc loss.')
    return parser.parse_args()

def main():

    train_losses = []
    val_losses = []
    args = parse_args()
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")   
        # random.seed(1111)
        # torch.manual_seed(1111)

    model = FasterRCNNResnet50()

    model.to(device)

    dataset_path = args.dataset_path

    data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = Dataset(dataset_path, phase='train', transform=data_transforms)
    print('Train set size:', train_set.__len__())
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    
    val_set = Dataset(dataset_path, phase='test', transform=data_transforms)
    print('Validation set size:', val_set.__len__())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=True, num_workers=args.workers)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Affectnet class weight
    # aff_weight = [74874, 134415, 25459, 14090, 6378, 3803, 24882]
    # raf-db class weight
    raf_weight = [1290, 281, 717, 4772, 1982, 705, 2524]
    class_counts = torch.from_numpy(
        np.array(raf_weight).astype(np.float32))
    class_weights = (torch.sum(class_counts) - class_counts) / class_counts

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.cuda(), reduction='none')
    
    best_acc = 0.
    best_std = 3

    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.
        correct_sum = 0
        iter_cnt = 0
        vlabels_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
        voutputs_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
        vloss_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}

        # training step
        with tqdm(total=int(train_set.__len__()) / args.bs) as pbar:
            for i, data in enumerate(training_loader):

                iter_cnt += 1
                idxs, imgs, orig_box_lsts, labels = data

                imgs = imgs.cuda()
                orig_box_lsts = orig_box_lsts.cuda()
                labels = labels.cuda()

                outputs = model(imgs, orig_box_lsts)
                _, predicts = torch.max(outputs, 1)
                _, loss_bmc = bmc_loss(predicts.unsqueeze(1), labels.unsqueeze(1))
                loss = loss_fn(outputs, labels).mean() + args.lamb * loss_bmc

                # Backropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # train loss
                # print(loss_fn(outputs, labels).mean(), supervised_contrastive_loss(outputs, labels))

                train_loss += loss.item()

                correct_num = torch.eq(predicts, labels)
                correct_sum += correct_num.sum().item()

                # progressbar
                pbar.set_description(f'TRAINING [{epoch+1}/{args.epochs}]')
                pbar.update(1)   
        
        train_acc = correct_sum / float(train_set.__len__())
        train_loss = train_loss/iter_cnt

        scheduler.step()

        

        # validation step
        with torch.no_grad():

            val_loss = 0.
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            with tqdm(total=int(val_set.__len__()) / args.bs) as pbar:
                for i, vdata in enumerate(val_loader):
                    idx, vimgs, vorig_box_lsts, vlabels = vdata
                    vimgs = vimgs.cuda()
                    vorig_box_lsts = vorig_box_lsts.cuda()
                    vlabels = vlabels.cuda()

                    voutputs = model(vimgs, vorig_box_lsts)
                    _, predicts = torch.max(voutputs, 1)
                    loss_bmc_item, loss_bmc = bmc_loss(predicts.unsqueeze(1), vlabels.unsqueeze(1))
                    vloss = loss_fn(voutputs, vlabels).mean() + args.lamb * loss_bmc
                    val_loss += vloss.item()
                    iter_cnt += 1

                    
                    correct_or_not = torch.eq(predicts, vlabels)
                    bingo_cnt += correct_or_not.sum().item()
                    
                    # val loss
                    # print(loss_fn(outputs, labels).mean(), supervised_contrastive_loss(outputs, labels))

                    vloss_array_ce = loss_fn(voutputs, vlabels).cpu().detach().numpy()
                    vloss_array_bmc = loss_bmc_item.cpu().detach().numpy()

                    # cal metrics
                    a_vlabels = vlabels.cpu().numpy()
                    a_predicts = predicts.cpu().numpy()
                    for a_i in range(len(a_vlabels)):
                        temp = vloss_array_ce[a_i] + args.lamb * vloss_array_bmc[a_i]
                        vloss_dict[str(a_vlabels[a_i])] += temp
                        vlabels_dict[str(a_vlabels[a_i])] += 1
                        if a_vlabels[a_i] == a_predicts[a_i]:
                            voutputs_dict[str(a_predicts[a_i])] += 1

                    # progressbar
                    pbar.set_description(f'VALIDATING [{epoch+1}/{args.epochs}]')
                    pbar.update(1)
                    
            val_loss = val_loss/iter_cnt
            val_acc = bingo_cnt / float(val_set.__len__())
            train_losses.append(np.around(train_loss, 4))
            val_losses.append(np.around(val_loss, 4))

            std, ser, expression_acc_dict, mean_acc = utils.cal_acc_per_expression(vlabels_dict, voutputs_dict)
            with open("log.txt", "a") as f:
                f.write("epoch: " + str(epoch) + " " + str(expression_acc_dict) + " val_acc: " +  str(val_acc) + " mean_acc: " + str(mean_acc) + " std: " + str(std) + "\n")
            print('[Epoch %d] Training loss: %.4f. Train accuracy: %.4f. Val loss: %.4f. Val accuracy: %.4f. Mean acc: %.4f' %
                    (epoch + 1, train_loss, train_acc, val_loss, val_acc, mean_acc))

            if val_acc > best_acc or std < best_std or (epoch+1 > (args.epochs - 5)): 
                if val_acc > best_acc:
                    best_acc = val_acc
                if std < best_std:
                    best_std = std
                print("best_acc: %.4f" % (best_acc), end=' ')
                print("STD: %.4f. SER: %.4f" % (std, ser))
                print(expression_acc_dict)
            # if val_acc > 0.835 and std < 0.095:    # raf-db
            # # if val_acc > 0.6 and std < 0.095:       # affectnet
            #     import os
            #     torch.save({'iter': epoch+1,
            #                 'model_state_dict': model.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict(), },
            #                 os.path.join("./models", "raf-db_" + str(epoch+1) + "_acc_" + str(val_acc) + "_std_" + str(std) + ".pth"))

            roi_weight = roi_weight_cal(vloss_dict, vlabels_dict)
        
        # extractor_roi.w reweight
        for name, p in model.named_parameters():
            if (name == 'extractor_roi.w'):

                p.data = torch.Tensor(roi_weight)

    print("best acc: %.4f. best std: %.4f" % (best_acc, best_std))
    utils.plot_curve(train_losses, val_losses)
    f.close()



def bmc_loss(pred, target):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    noise_sigma = torch.nn.Parameter(torch.tensor(0.05))
    noise_var = noise_sigma ** 2
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss_temp = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda(), reduction='none')     # contrastive-like loss
    loss = loss_temp.mean() * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss_temp, loss


def roi_weight_cal(vloss_dict, vlabels_dict):

    roi_weight = [0, 0, 0, 0, 0, 0, 0]
    # neutral not contains au
    ex_au_group_dict = {'0': [2, 4], '1': [0,5], '2': [0, 1, 4], '3': [0, 1, 6], '4': [0, 1, 5], '5': [3, 4, 5]}  # raf-db
    # ex_au_group_dict = {'1': [0, 1, 6], '2': [0, 1, 5], '3': [2, 4], '4': [0,5], '5': [0, 1, 4], '6': [3, 4, 5]}  # affectnet
    au_group_list = []
    vloss_sum = 0.
    for i in range(0, 7):
        # if i != 6:
        temp = int(vloss_dict[str(i)]) / int(vlabels_dict[str(i)])
        vloss_sum += temp
    mean_vloss = vloss_sum / 7
    for i in range(0, 7):
        if  int(vloss_dict[str(i)]) / int(vlabels_dict[str(i)]) > mean_vloss and i != 6:
            au_group_list.append(ex_au_group_dict[str(i)])
    # # Union
    au_group_union = [i for item in au_group_list for i in item]
    for i in range(len(au_group_union)):
        roi_weight[au_group_union[i]] += 1

    return roi_weight



if __name__ == "__main__":
    args = parse_args()
    main()
        


