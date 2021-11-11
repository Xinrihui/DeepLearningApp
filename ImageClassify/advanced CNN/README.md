
# 高级的图像分类模型

## 1.使用 Keras 实现 ResNet 模型

### 1.1 模型设计

    参考论文: Deep Residual Learning for Image Recognition

### 1.2 实验结果

      1.数据集 signs
        n_train = 1080
        n_test = 120
        使用 Adam 梯度下降
        mini_batch_size = 32
        num_epochs=20
        test accuracy：0.97
        train accuracy：0.98

## 2.使用 Keras 实现 Inception 模型

### 2.1 模型设计

    参考论文: Going Deeper with Convolutions

### 2.2 实验结果

      1.数据集 signs
        n_train = 1080
        n_test = 120
        使用 Adam 梯度下降
        mini_batch_size = 32
        num_epochs=20
        test accuracy： 0.99
        train accuracy：1.00

      2.数据集 cifar-10
        n_train = 49000
        n_val = 1000
        n_test = 1000
        使用 Adam 梯度下降
        mini_batch_size = 512
        num_epochs=30
        test accuracy： 0.77
        train accuracy：1.00

## Ref



## Notes

    1. cifar-10 数据集获取  http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

