import numpy as np

import matplotlib.pyplot as plt

# raf-db 
expression = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
# affectnet
# expression = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

def cal_acc_per_expression(vlabels_dict, voutputs_dict):
    expression_acc_dict = dict()
    temp = []
    count = 0
    for _ in range(0, 7):
        acc = float(voutputs_dict[str(_)]) / float(vlabels_dict[str(_)])
        acc = np.around(acc, 4)
        count += acc
        temp.append(acc)
        expression_acc_dict[str(expression[_])] = acc

    mean_acc = np.around(count / 7, 4)
    std = np.std(temp, ddof=0)
    ser = (1 - min(temp)) / (1 - max(temp))

    return std, ser, expression_acc_dict, mean_acc

def plot_curve(train_losses, val_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig("Loss_Curve.jpg")