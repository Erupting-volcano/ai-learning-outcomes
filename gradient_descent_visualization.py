import numpy as np
import matplotlib.pyplot as plt
def price(w,b,x,y):
    return (w*x+b-y)**2

def main():
    n=100000#循环次数
    w=1#初始化w
    b=1#初始化b
    a=0.001#学习率
    loss_history = []
    w_history = []
    b_history = []
    #arr=np.array([[1,1],[2,2],[3,3]],dtype=np.float64)
    arr = np.array([
        [1, 4.9],
        [2, 7.1],
        [3, 9.2],
        [4, 11.0],
        [5, 12.7],
        [6, 15.3]
    ], dtype=np.float64)
    for i in range(n):

        for x,y in arr:#反向传播
            dw=2*x*(w*x+b-y)
            db=2*(w*x+b-y)
            w=w-a*dw
            b=b-a*db
        loss=0
        for x,y in arr:#计算loss
            loss+=price(w,b,x,y)
        loss /= arr.shape[1]
        loss_history.append(loss)
        w_history.append(w)
        b_history.append(b)
        if i%10==0:
            print(f"第{i}轮 | w={w:.4f} | b={b:.4f} | 平均loss={loss:.6f}")
    x_data = arr[:, 0]
    y_data = arr[:, 1]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)  # 把画布分成1行3列，画第1张图
    plt.scatter(x_data, y_data)  # 画散点（原始数据）
    plt.plot(x_data, w * x_data + b)  # 画直线（模型拟合结果）
    plt.subplot(1, 3, 2)
    plt.plot(loss_history)  # 把loss列表的数据连成线
    plt.subplot(1, 3, 3)
    plt.plot(w_history)  # 画w的变化
    plt.plot(b_history)  # 画b的变化
    plt.axhline(y=1.0, linestyle='--')  # 画w的目标值虚线
    plt.show()
if __name__ == '__main__':
    main()