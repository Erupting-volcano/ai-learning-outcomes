import numpy as np
import pathlib
import matplotlib.pyplot as plt

class Dataloader:#数据读取
    def get_data(self):
        img=np.fromfile("./data/train-images.idx3-ubyte", np.uint8, offset=16)#导入image数据
        img = img.reshape(-1, 784)#将一维数据分成60000份图片
        img = img.astype(np.float32) / 255#归一化
        lab = np.fromfile("./data/train-labels.idx1-ubyte", np.uint8, offset=8)#导入labels数据
        lab = np.eye(10)[lab]#标签矩阵化
        return img, lab
class TextDataloader:#测试数据读取
    def get_data(self):
        img=np.fromfile("./data/t10k-images.idx3-ubyte", np.uint8, offset=16)#导入image数据
        img = img.reshape(-1, 784)#将一维数据分成60000份图片
        img = img.astype(np.float32) / 255#归一化
        lab = np.fromfile("./data/t10k-labels.idx1-ubyte", np.uint8, offset=8)#导入labels数据
        lab = np.eye(10)[lab]#标签矩阵化
        return img, lab


if __name__ == "__main__":
    # 通过dataloader读取数据
    dataloader = Dataloader()
    images, labels = dataloader.get_data()
    #设置w,b矩阵
    w_1=np.random.uniform(-0.5, 0.5, (20, 784))
    w_2=np.random.uniform(-0.5, 0.5, (10, 20))
    b_1=np.zeros((20, 1))
    b_2=np.zeros((10, 1))
    #设置超参数
    learn_rate = 0.03#学习率
    nr_correct = 0#本轮成功次数
    epochs = 5#训练轮数
    epochs_index=0#已经训练的轮数
    for epoch in range(epochs):
        sum_loss = 0
        for i in range(images.shape[0]):#遍历训练集所有数据
            img_ = images[i].reshape(-1, 1)
            lab_ = labels[i].reshape(-1, 1)
            #前向传播
            z1=b_1+w_1@img_
            a1=1 / (1 + np.exp(-z1))
            z2=b_2+w_2@a1
            a2=1 / (1 + np.exp(-z2))
            #本次损失函数
            loss=np.sum((a2-lab_)**2)
            sum_loss+=loss
            #反向传播
            delta2 = (a2 - lab_) * (a2 * (1 - a2))#loss到2层的导数
            delta1 = (w_2.T @ delta2) * (a1 * (1 - a1))  # loss到一层的导数
            w_2 -= learn_rate * delta2 @ a1.T#更新2层
            b_2 -= learn_rate * delta2
            w_1 -= learn_rate * delta1 @ img_.T#更新一层
            b_1 -= learn_rate * delta1
            if np.argmax(a2) == np.argmax(lab_):# 统计正确预测个数
                nr_correct += 1
        epochs_index += 1
        print(f"第{epochs_index}轮，平均损失：{sum_loss/images.shape[0]:.4f},正确率：{nr_correct/images.shape[0]:.4f}")
        nr_correct=0
###############################################神经网络训练################################################



    # 通过TextDataloader读取数据
    textdataloader = TextDataloader()
    images_t, labels_t = textdataloader.get_data()
    sum_loss_t=0
    nr_correct_t=0
    for i in range(images_t.shape[0]):
        img_t = images_t[i].reshape(-1, 1)
        lab_t = labels_t[i].reshape(-1, 1)
        # 前向传播
        z1_t = b_1 + w_1 @ img_t
        a1_t = 1 / (1 + np.exp(-z1_t))
        z2_t = b_2 + w_2 @ a1_t
        a2_t = 1 / (1 + np.exp(-z2_t))
        # 本次损失函数
        loss_t = np.sum((a2_t - lab_t) ** 2)
        sum_loss_t += loss_t
        if np.argmax(a2_t) == np.argmax(lab_t):# 统计正确预测个数
            nr_correct_t += 1
    print(f"平均损失：{sum_loss_t/images_t.shape[0]:.4f},正确率：{nr_correct_t/images_t.shape[0]:.4f}")
