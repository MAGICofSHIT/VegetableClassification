import matplotlib.pyplot as plt
import re

# 读取数据
data = """
Epoch 1/60
----------
Val Phase: Train Loss: 0.7036 Acc: 0.8593
Train Phase: Val Loss: 0.1114 Acc: 0.9887
Epoch 2/60
----------
Val Phase: Train Loss: 0.2123 Acc: 0.9573
Train Phase: Val Loss: 0.0638 Acc: 0.9903
Epoch 3/60
----------
Val Phase: Train Loss: 0.1601 Acc: 0.9630
Train Phase: Val Loss: 0.0420 Acc: 0.9930
Epoch 4/60
----------
Val Phase: Train Loss: 0.1370 Acc: 0.9667
Train Phase: Val Loss: 0.0356 Acc: 0.9927
Epoch 5/60
----------
Val Phase: Train Loss: 0.1223 Acc: 0.9681
Train Phase: Val Loss: 0.0283 Acc: 0.9950
Epoch 6/60
----------
Val Phase: Train Loss: 0.1106 Acc: 0.9714
Train Phase: Val Loss: 0.0236 Acc: 0.9963
Epoch 7/60
----------
Val Phase: Train Loss: 0.0969 Acc: 0.9734
Train Phase: Val Loss: 0.0228 Acc: 0.9967
Epoch 8/60
----------
Val Phase: Train Loss: 0.0961 Acc: 0.9734
Train Phase: Val Loss: 0.0214 Acc: 0.9953
Epoch 9/60
----------
Val Phase: Train Loss: 0.0909 Acc: 0.9749
Train Phase: Val Loss: 0.0220 Acc: 0.9940
Epoch 10/60
----------
Val Phase: Train Loss: 0.0899 Acc: 0.9743
Train Phase: Val Loss: 0.0172 Acc: 0.9970
Epoch 11/60
----------
Val Phase: Train Loss: 0.0841 Acc: 0.9768
Train Phase: Val Loss: 0.0171 Acc: 0.9967
Epoch 12/60
----------
Val Phase: Train Loss: 0.0805 Acc: 0.9771
Train Phase: Val Loss: 0.0172 Acc: 0.9963
Epoch 13/60
----------
Val Phase: Train Loss: 0.0748 Acc: 0.9785
Train Phase: Val Loss: 0.0167 Acc: 0.9967
Epoch 14/60
----------
Val Phase: Train Loss: 0.0793 Acc: 0.9782
Train Phase: Val Loss: 0.0170 Acc: 0.9967
Epoch 15/60
----------
Val Phase: Train Loss: 0.0768 Acc: 0.9785
Train Phase: Val Loss: 0.0168 Acc: 0.9967
Epoch 16/60
----------
Val Phase: Train Loss: 0.0810 Acc: 0.9766
Train Phase: Val Loss: 0.0166 Acc: 0.9967
Epoch 17/60
----------
Val Phase: Train Loss: 0.0720 Acc: 0.9798
Train Phase: Val Loss: 0.0165 Acc: 0.9963
Epoch 18/60
----------
Val Phase: Train Loss: 0.0789 Acc: 0.9774
Train Phase: Val Loss: 0.0160 Acc: 0.9970
Epoch 19/60
----------
Val Phase: Train Loss: 0.0807 Acc: 0.9771
Train Phase: Val Loss: 0.0160 Acc: 0.9970
Epoch 20/60
----------
Val Phase: Train Loss: 0.0816 Acc: 0.9770
Train Phase: Val Loss: 0.0156 Acc: 0.9967
Epoch 21/60
----------
Val Phase: Train Loss: 0.0745 Acc: 0.9796
Train Phase: Val Loss: 0.0156 Acc: 0.9970
Epoch 22/60
----------
Val Phase: Train Loss: 0.0770 Acc: 0.9787
Train Phase: Val Loss: 0.0156 Acc: 0.9970
Epoch 23/60
----------
Val Phase: Train Loss: 0.0751 Acc: 0.9785
Train Phase: Val Loss: 0.0151 Acc: 0.9970
Epoch 24/60
----------
Val Phase: Train Loss: 0.0753 Acc: 0.9797
Train Phase: Val Loss: 0.0158 Acc: 0.9967
Epoch 25/60
----------
Val Phase: Train Loss: 0.0758 Acc: 0.9790
Train Phase: Val Loss: 0.0162 Acc: 0.9967
Epoch 26/60
----------
Val Phase: Train Loss: 0.0746 Acc: 0.9793
Train Phase: Val Loss: 0.0152 Acc: 0.9973
Epoch 27/60
----------
Val Phase: Train Loss: 0.0706 Acc: 0.9813
Train Phase: Val Loss: 0.0156 Acc: 0.9970
Epoch 28/60
----------
Val Phase: Train Loss: 0.0768 Acc: 0.9777
Train Phase: Val Loss: 0.0157 Acc: 0.9967
Epoch 29/60
----------
Val Phase: Train Loss: 0.0755 Acc: 0.9790
Train Phase: Val Loss: 0.0154 Acc: 0.9970
Epoch 30/60
----------
Val Phase: Train Loss: 0.0743 Acc: 0.9792
Train Phase: Val Loss: 0.0159 Acc: 0.9967
Epoch 31/60
----------
Val Phase: Train Loss: 0.0766 Acc: 0.9778
Train Phase: Val Loss: 0.0157 Acc: 0.9963
Epoch 32/60
----------
Val Phase: Train Loss: 0.0692 Acc: 0.9809
Train Phase: Val Loss: 0.0155 Acc: 0.9967
Epoch 33/60
----------
Val Phase: Train Loss: 0.0768 Acc: 0.9785
Train Phase: Val Loss: 0.0159 Acc: 0.9967
Epoch 34/60
----------
Val Phase: Train Loss: 0.0795 Acc: 0.9771
Train Phase: Val Loss: 0.0155 Acc: 0.9970
Epoch 35/60
----------
Val Phase: Train Loss: 0.0735 Acc: 0.9789
Train Phase: Val Loss: 0.0158 Acc: 0.9967
Epoch 36/60
----------
Val Phase: Train Loss: 0.0794 Acc: 0.9769
Train Phase: Val Loss: 0.0157 Acc: 0.9967
Epoch 37/60
----------
Val Phase: Train Loss: 0.0753 Acc: 0.9776
Train Phase: Val Loss: 0.0155 Acc: 0.9967
Epoch 38/60
----------
Val Phase: Train Loss: 0.0776 Acc: 0.9785
Train Phase: Val Loss: 0.0153 Acc: 0.9970
Epoch 39/60
----------
Val Phase: Train Loss: 0.0778 Acc: 0.9766
Train Phase: Val Loss: 0.0161 Acc: 0.9967
Epoch 40/60
----------
Val Phase: Train Loss: 0.0710 Acc: 0.9793
Train Phase: Val Loss: 0.0158 Acc: 0.9967
Epoch 41/60
----------
Val Phase: Train Loss: 0.0756 Acc: 0.9796
Train Phase: Val Loss: 0.0156 Acc: 0.9970
Epoch 42/60
----------
Val Phase: Train Loss: 0.0722 Acc: 0.9799
Train Phase: Val Loss: 0.0155 Acc: 0.9967
Epoch 43/60
----------
Val Phase: Train Loss: 0.0738 Acc: 0.9791
Train Phase: Val Loss: 0.0152 Acc: 0.9970
Epoch 44/60
----------
Val Phase: Train Loss: 0.0765 Acc: 0.9788
Train Phase: Val Loss: 0.0160 Acc: 0.9967
Epoch 45/60
----------
Val Phase: Train Loss: 0.0706 Acc: 0.9799
Train Phase: Val Loss: 0.0155 Acc: 0.9967
Epoch 46/60
----------
Val Phase: Train Loss: 0.0717 Acc: 0.9796
Train Phase: Val Loss: 0.0154 Acc: 0.9970
Epoch 47/60
----------
Val Phase: Train Loss: 0.0705 Acc: 0.9804
Train Phase: Val Loss: 0.0158 Acc: 0.9967
Epoch 48/60
----------
Val Phase: Train Loss: 0.0723 Acc: 0.9799
Train Phase: Val Loss: 0.0153 Acc: 0.9970
Epoch 49/60
----------
Val Phase: Train Loss: 0.0726 Acc: 0.9794
Train Phase: Val Loss: 0.0153 Acc: 0.9967
Epoch 50/60
----------
Val Phase: Train Loss: 0.0700 Acc: 0.9800
Train Phase: Val Loss: 0.0157 Acc: 0.9970
Epoch 51/60
----------
Val Phase: Train Loss: 0.0715 Acc: 0.9800
Train Phase: Val Loss: 0.0156 Acc: 0.9967
Epoch 52/60
----------
Val Phase: Train Loss: 0.0701 Acc: 0.9800
Train Phase: Val Loss: 0.0153 Acc: 0.9970
Epoch 53/60
----------
Val Phase: Train Loss: 0.0699 Acc: 0.9798
Train Phase: Val Loss: 0.0154 Acc: 0.9970
Epoch 54/60
----------
Val Phase: Train Loss: 0.0720 Acc: 0.9795
Train Phase: Val Loss: 0.0155 Acc: 0.9967
Epoch 55/60
----------
Val Phase: Train Loss: 0.0708 Acc: 0.9800
Train Phase: Val Loss: 0.0157 Acc: 0.9967
Epoch 56/60
----------
Val Phase: Train Loss: 0.0705 Acc: 0.9797
Train Phase: Val Loss: 0.0156 Acc: 0.9967
Epoch 57/60
----------
Val Phase: Train Loss: 0.0725 Acc: 0.9792
Train Phase: Val Loss: 0.0154 Acc: 0.9970
Epoch 58/60
----------
Val Phase: Train Loss: 0.0700 Acc: 0.9799
Train Phase: Val Loss: 0.0153 Acc: 0.9970
Epoch 59/60
----------
Val Phase: Train Loss: 0.0712 Acc: 0.9796
Train Phase: Val Loss: 0.0155 Acc: 0.9967
Epoch 60/60
----------
Val Phase: Train Loss: 0.0699 Acc: 0.9800
Train Phase: Val Loss: 0.0154 Acc: 0.9970
"""

train_loss = []
train_acc = []
val_loss = []
val_acc = []

# 解析数据
for line in data.strip().split('\n'):
    train_match = re.search(r'Val Phase: Train Loss: ([\d.]+) Acc: ([\d.]+)', line)
    val_match = re.search(r'Train Phase: Val Loss: ([\d.]+) Acc: ([\d.]+)', line)

    if train_match:
        train_loss.append(float(train_match.group(1)))
        train_acc.append(float(train_match.group(2)))
    elif val_match:
        val_loss.append(float(val_match.group(1)))
        val_acc.append(float(val_match.group(2)))

# 输出调试信息
print(f'Train Loss: {train_loss}')
print(f'Train Accuracy: {train_acc}')
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

# 确保 val_loss 和 val_acc 不是空列表
if not val_loss or not val_acc:
    print("Validation loss or accuracy data is missing. Please check the input data format.")
    exit()

# 绘图
epochs = list(range(1, len(train_loss) + 1))

# 绘制损失曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Train Loss', color='green')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('loss_over_epochs.png')  # 保存损失图像
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('accuracy_over_epochs.png')  # 保存准确率图像
plt.show()