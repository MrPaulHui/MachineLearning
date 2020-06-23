# 线性回归、交叉熵与似然函数

## 线性回归

设数据集为$X\in R^{D*N}$，N为样本数，D为数据维数，对应的label为$Y\in R^{1*N}$

线性回归模型表达式：
$$
h_\theta(X)=\theta^TX
$$
写成标量形式：
$$
h_\theta(x_i)=\sum_{j=1}^D\theta_jx_{ij}
$$
$x_i\in R^{D}$，为X的列向量

其中，$\theta\in R^{D*1}$，为模型参数

注：一般还会有一个偏置参数b加在后面，不过偏置可以放到$\theta$里面，加一列b的列向量，同时在X下面加一行全为1的行向量即可，所以一般形式就写成上式即可。

有两个角度可以推出线性回归模型的优化目标：

1. 最小二乘法（L2 loss）
2. 极大似然估计

求解优化目标有两种方法：

1. 梯度下降法
2. 正规方程法（广义逆）

### 推导优化目标

#### 1. 最小二乘法（L2 loss）

模型输出和label的误差的平方和最小。

优化目标（损失函数）
$$
\min_\theta J(\theta)=\frac{1}{N}\sum_{i=i}^N(y_i-h_\theta(x_i))^2
$$
写成矩阵形式为
$$
\min_\theta J(\theta)=\frac{1}{N}||Y-h_\theta(X)||_2^2
$$

#### 2. 极大似然估计

参考：https://zhuanlan.zhihu.com/p/36331115

极大似然估计参考：https://blog.csdn.net/u011508640/article/details/72815981

设线性回归模型的输出与实际label的误差为$\epsilon=y_i-h_\theta(x_i)$，误差值在0周围分布，假设误差值服从正太分布
$$
\epsilon\sim N(0,\sigma^2)
$$
则，label $y_i$服从的分布为
$$
y_i\sim N(h_\theta(x_i),\sigma^2)
$$
所以，Y的似然函数为
$$
P(Y|X,\theta)=\prod_{i=1}^N\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-h_\theta(x_i))^2}{2\sigma^2})
$$
对数似然函数为
$$
L(\theta)=\log P(Y|X,\theta)=-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-h_\theta(x_i))^2+N\log\frac{1}{\sqrt{2\pi}\sigma}
$$
目标是最大化似然函数，即
$$
\max_\theta L(\theta)=\max_\theta -\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-h_\theta(x_i))^2
$$
等价于
$$
\min_\theta\sum_{i=1}^N(y_i-h_\theta(x_i))^2
$$
可以发现和最小二乘法（L2 loss）的优化目标（损失函数）是等价的。

后面求解优化问题方法就和L2 loss一致了。

#### 极大似然估计和最小二乘法的联系

最小二乘法是求模型参数，使得模型输出尽可能的去接近label，二者误差的平方和最小。

极大似然估计是求模型参数，使得模型输出的分布尽可能接近观测到的分布（label）。

二者出发点都是使模型输出尽可能接近label，所用方法不同，殊途同归。

### 求解优化目标

#### 1. 梯度下降法

#### 2. 正规方程（广义逆）求解优化目标

参考：https://zhuanlan.zhihu.com/p/34842727

损失函数的矩阵形式为
$$
J(\theta)=\frac{1}{N}||Y-h_\theta(X)||_2^2\\
=\frac{1}{N}(Y-h_\theta(X))(Y-h_\theta(X))^T\\
=\frac{1}{N}(Y-\theta^TX)(Y-\theta^TX)^T\\
=\frac{1}{N}(YY^T-YX^T\theta-\theta^TXY^T+\theta^TXX^T\theta)
$$
对$\theta$求导，得
$$
\frac{\part J(\theta)}{\part \theta}=\frac{1}{N}(-XY^T-XY^T+2XX^T\theta)\\
=\frac{2}{N}(-XY^T+XX^T\theta)
$$
令导数为0，所以
$$
\theta=(XX^T)^{-1}XY^T
$$
可直接根据数据X和label Y得到模型参数。

## 交叉熵与极大似然估计的联系

从极大似然估计的角度推导交叉熵损失

### 二分类

参考：https://zhuanlan.zhihu.com/p/36331115

设数据集为$X\in R^{N*D}$，对应的label为$y_i\in R, y_i\in\{0,1\}$

分类模型为$f_\theta(x_i)\in R$，输出值表示样本类别为1的概率，一般为sigmoid函数做分类层。

似然函数为
$$
P(Y|X,\theta)=\prod_{i=1}^N[f_\theta(x_i)]^{y_i}[1-f_\theta(x_i)]^{1-y_i}
$$
对数似然函数为
$$
L(\theta)=\log P(Y|X,\theta)=\sum_{i=1}^N[y_i\log f_\theta(x_i)+(1-y_i)\log (1-f_\theta(x_i))]
$$
目标是最大化似然函数，即
$$
\max_\theta L(\theta)=\max_\theta\sum_{i=1}^N[y_i\log f_\theta(x_i)+(1-y_i)\log (1-f_\theta(x_i))]
$$
等价于
$$
\min_\theta\sum_{i=1}^N[-y_i\log f_\theta(x_i)-(1-y_i)\log (1-f_\theta(x_i))]
$$
即推出了二分类的交叉熵损失函数。

### 多分类

设数据集为$X\in R^{N*D}$，对应的label为$y_i\in R^C$，$y_i=[y_i^{(1)},y_i^{(2)},...,y_i^{(C)}]$，其中只有一维为1，其余都为0

分类模型为$f_\theta(x_i)\in R^C$，$f_\theta(x_i)=[f_\theta^{(1)}(x_i),f_\theta^{(2)}(x_i),...,f_\theta^{(C)}(x_i)]$，输出值的每一维表示样本属于对应类别的概率，且$\sum_{j=1}^Cf_\theta^{(j)}(x_i)=1$，一般为softmax函数做分类层。

似然函数为
$$
P(Y|X,\theta)=\prod_{i=1}^N\prod_{j=1}^C [f_\theta^{(j)}(x_i)]^{y_i^{(j)}}
$$
对数似然函数为
$$
L(\theta)=\log P(Y|X,\theta)=\sum_{i=1}^N(\sum_{j=1}^Cy_i^{(j)}\log f_\theta^{(j)}(x_i))
$$
目标是最大化似然函数，即
$$
\max_\theta L(\theta)=\max_\theta\sum_{i=1}^N(\sum_{j=1}^Cy_i^{(j)}\log f_\theta^{(j)}(x_i))
$$
等价于
$$
\min_\theta\sum_{i=1}^N(-\sum_{j=1}^Cy_i^{(j)}\log f_\theta^{(j)}(x_i))
$$
即推出了多分类的交叉熵损失函数。



