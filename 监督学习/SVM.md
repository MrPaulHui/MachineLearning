# SVM

注：本文公式中的\*都是表示內积。

## SVM核心思想——间隔最大化

<img src="https://img-blog.csdn.net/20140829135959290" alt="img" style="zoom:50%;" />

对于一个分类任务，需要在特征空间中寻找一个分类面，将正类和负类分开，如何找这个分类面，就是机器学习要做的事。

对于分类面，除了要求其能正确分类样本，还需要其**间隔**尽可能的大，间隔就是分类面距离最近的样本的距离，只有这个距离大，才可以使模型对最难分类的样本也可以有足够大的确信度进行分类，才可以使模型具有鲁棒性。如果这个间隔过小，非常靠近样本，那么新的测试样本发生了一点变动，就有可能偏移到分类面的另一侧去，导致分类错误。（也可以说是抑制模型过拟合，和人脸识别里加margin的insight是大致一致的）

所以使得**间隔最大化就是SVM支持向量机的核心目标**。首先需要对间隔进行具体的定义。

设训练集为$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$，其中$x_i\in R^D,y_i\in \{+1,-1\}$，有N个样本，每个样本有D维特征。

设分类超平面为$w*x+b=0$，对应的分类决策函数为$f(x) =sign(w*x+b)$

### 函数间隔与几何间隔

样本点$(x_i,y_i)$到超平面$w*x+b=0$的距离公式为
$$
Dist_i=\frac{|w*x_i+b|}{||w||}
$$
其中，$w*x_i+b$与$y_i$符号是否一致，决定了分类是否正确，所以可以用
$$
\gamma_i=y_i\frac{w*x_i+b}{||w||}
$$
表示距离，同时表示分类是否正确。这就是几何间隔。

定义超平面$w*x+b=0$关于训练集$T$的几何间隔为所有样本点关于超平面$w*x+b=0$的几何间隔的最小值
$$
\gamma=\min_{i=1,2,...,N}\gamma_i
$$
至于函数间隔，把几何间隔的分母$||w||$去掉，就是函数间隔了，即函数间隔为
$$
\gamma_i=y_i(w*x_i+b)
$$
函数间隔和几何间隔关系为
$$
\hat \gamma_i=\gamma_i*||w||\\
\hat \gamma = \gamma*||w||
$$
不过没有啥意义啊，最终用的间隔也是几何间隔。

## 线性可分SVM

线性可分的意思是，对于一个训练集，可以用一个超平面将其所有的样本进行正确分类。

### 优化目标

根据SVM间隔最大化的核心，**线性可分SVM的优化目标为：在保证所有样本正确分类的基础上，使得分类面关于训练集的几何间隔最大**。（这也是为什么对于一个线性可分数据集，分类面可以有无穷个，但满足SVM的只有一个）。即
$$
\max_{w,b}\ \gamma \\
s.t. \ y_i\frac{w*x_i+b}{||w||}\geq \gamma,\ \ i=1,2,...,N
$$
考虑到函数间隔和几何间隔的关系，有
$$
\max_{w,b}\ \frac{\hat \gamma}{||w||} \\
s.t. \ y_i(w*x_i+b)\geq \hat\gamma,\ \ i=1,2,...,N
$$
若令$w=\lambda w,b=\lambda b$，则$\hat\gamma=\lambda\hat\gamma$，代入到优化式子，发现没有任何变化，所以不妨取$\hat\gamma=1$，同时有最大化$\frac{1}{||w||}$等价于最小化$\frac{1}{2}||w||^2$，所以优化式可以写成
$$
\min_{w,b}\ \frac{1}{2}||w||^2\\
s.t. \ y_i(w*x_i+b)-1\geq 0, \ \ i=1,2,...,N
$$
求解这个优化式，得到$w^*,b^*$，进而得到分类面为
$$
w^**x+b^*=0
$$
决策函数为
$$
f(x)=sign(w^**x+b^*)
$$

#### 支持向量

支持向量的意思就是距离分类面最近的点，也就是约束条件取等的点，即
$$
y_i(w*x_i+b)-1=0
$$
对于$y_i=+1$的正类样本，支持向量在超平面$H_1:w*x+b=1$上，与分类超平面的距离为$\frac{1}{||w||}$

对于$y_i=-1$的正类样本，支持向量在超平面$H_2:w*x+b=-1$上，与分类超平面的距离为$\frac{1}{||w||}$

$H_1,H_2$称作间隔边界。

可以发现，$H_1,H_2$平行，且二者之间没有其他的样本点，二者的距离为$\frac{2}{||w||}$

注意，**对于线性可分SVM来说，分类面一定位于两个间隔边界的中央**。假设分类面不在中央，朝某一侧的间隔边界偏移了，那么几何间隔就会变小，不满足间隔最大化的要求。

从两个间隔边界的距离角度看优化式子：间隔最大化等价于两个间隔边界的距离最大化，就是最大化$\frac{ 1}{||w||}$

### 求解方法

求解的核心方法是利用**拉格朗日对偶性**。

拉格朗日对偶性具体算法可参考《统计学习方法》附录C。下面根据拉格朗日对偶性对线性可分SVM优化目标进行求解：

#### 1. 构造拉格朗日函数

$$
L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^N\alpha_i[1-y_i(w*x_i+b)]\\
=\frac{1}{2}||w||^2+\sum_{i=1}^N\alpha_i-\sum_{i=1}^N\alpha_iy_i(w*x_i+b)
$$

其中，$\alpha_i\geq 0$

原始的线性可分SVM优化问题就等价于
$$
\min_{w,b}\max_{\alpha;\alpha_i\geq0}L(w,b,\alpha)
$$
这里的证明参考《统计学习方法》附录C。

#### 2. 应用拉格朗日对偶性构造对偶问题并求解

$\min_{w,b}\max_{\alpha;\alpha_i\geq0}L(w,b,\alpha)$的对偶问题为
$$
\max_{\alpha;\alpha_i\geq0}\min_{w,b}L(w,b,\alpha)
$$
先求解这个对偶问题

(1)求$\min_{w,b}L(w,b,\alpha)$
$$
\min_{w,b}L(w,b,\alpha)=\min_{w,b}\frac{1}{2}||w||^2+\sum_{i=1}^N\alpha_i-\sum_{i=1}^N\alpha_iy_i(w*x_i+b)
$$

$$
0=\frac{\part L}{\part w}=w-\sum_{i=1}^N\alpha_iy_ix_i\\
0=\frac{\part L}{\part b}=-\sum_{i=1}^N\alpha_iy_i
$$

得到，
$$
w=\sum_{i=1}^N\alpha_iy_ix_i\\
\sum_{i=1}^N\alpha_iy_i=0
$$
代入$L$表达式，有
$$
L(w,b,\alpha)=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)+\sum_{i=1}^N\alpha_i-\sum_{i=1}^N\alpha_iy_i((\sum_{j=1}^N\alpha_jy_jx_j)*x_i+b)\\
=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)+\sum_{i=1}^N\alpha_i
$$
即
$$
\min_{w,b}L(w,b,\alpha)=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)+\sum_{i=1}^N\alpha_i
$$
(2)求$\max_{\alpha;\alpha_i\geq0}\min_{w,b}L(w,b,\alpha)$

即求解如下的约束优化问题：
$$
\max_{\alpha}-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)+\sum_{i=1}^N\alpha_i\\
s.t.\ \sum_{i=1}^N\alpha_iy_i=0\\
s.t.\ \alpha_i\geq0,\ i=1,2,...,N
$$
将极大转换为极小，最终形式为
$$
\min_{\alpha}\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \sum_{i=1}^N\alpha_iy_i=0\\
s.t.\ \alpha_i\geq0,\ i=1,2,...,N
$$

#### 3. 根据拉格朗日对偶性判定定理推出对偶问题的解就是原始问题的解

根据《统计学习方法》附录C.2定理，可以判定上面所求的对偶问题的解就是原始线性可分SVM优化问题的解。

假设对偶问题的解为$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$

#### 4. 根据对偶问题的解和KKT条件求出原始问题的解

假设原始问题的解为$w^*,b^*$，那么$\alpha^*$是对偶问题的解且$w^*,b^*$是原始问题的解的充分必要条件是$w^*,b^*,\alpha^*$满足如下的KKT条件：
$$
\frac{\part L(w^*,b^*,\alpha^*)}{\part w}=w^*-\sum_{i=1}^N\alpha^*y_ix_i=0\\
\frac{\part L(w^*,b^*,\alpha^*)}{\part b}=-\sum_{i=1}^N\alpha_i^*y_i=0\\
\alpha_i^*(y_i(w^**x_i+b^*)-1)=0,\ i=1,2,...,N\\
y_i(w^**x_i+b^*)-1\geq0,\ i=1,2,...,N\\
\alpha_i^*\geq0,\ i=1,2,...,N
$$
所以，得到
$$
w^*=\sum_{i=1}^N\alpha_i^*y_ix_i
$$
对$\alpha^*$，至少有一个$\alpha_j^*>0$，这个结论可以用反证法来证明：若$\alpha^*=0$，则$w^*=0$，但$w^*=0$不是最优化的解，所以矛盾。对这个j，有
$$
y_j(w^**x_j+b^*)-1=0
$$
代入$w^*$的表达式，有
$$
b^*=y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i*x_j)
$$
即得到了最终的线性可分SVM的解。

最终的分类决策函数为
$$
f(x)=sign(\sum_{i=1}^N\alpha_i^*y_ix_i*x+y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i*x_j))
$$

## 线性SVM

线性不可分的意思是，对于一个训练集，任意一个超平面都无法将其所有样本完全正确分类，总存在一些特异点（outlier），如果把这些特异点去掉，那么剩下的样本组成的数据集是线性可分的。

对于不满足线性可分的数据集，线性可分SVM是不适用的，因为约束条件必然有样本点无法满足。因此需要有一个更一般的方法。

### 优化目标

线性不可分意味着训练集中某些样本点$(x_i,y_i)$不能满足约束条件$y_i(w*x_i+b)-1\geq 0$，不能满足是因为这些特异点分类错了，即$y_i(w*x_i+b)<0$，要想使这些点也满足约束条件，就需要加上一个松弛变量$\zeta_i\geq 0$，即约束条件变为
$$
y_i(w*x_i+b)\geq 1-\zeta_i
$$
对于这个松弛变量，当然是希望其越小越好，如果不限制其大小，每个样本点都可能有一个很大的松弛变量，这就导致大量的样本被错误分类，是不可接受的。所以需要在优化目标里极小化所有样本点松弛变量的和，再加上原来的优化目标，即得到
$$
\min_{w,b,\zeta}\ \frac{1}{2}||w||^2+C\sum_{i=1}^N \zeta_i
$$
其中，$C$为惩罚参数，用来调整两个优化目标各自的重要程度。

综上，线性SVM的优化目标为
$$
\min_{w,b}\ \frac{1}{2}||w||^2+C\sum_{i=1}^N\zeta_i \\
s.t. \ y_i(w*x_i+b)\geq 1-\zeta_i, \ \ i=1,2,...,N \\
s.t. \ \zeta_i\geq 0,\ \ i=1,2,...,N
$$
显然，线性SVM包括线性可分SVM。

#### 线性SVM推出hinge loss以及多分类的SVM loss

推出的线性SVM的约束条件
$$
y_i(w*x_i+b)\geq 1-\zeta_i, \ \ i=1,2,...,N
$$
可以写成
$$
\zeta_i\geq 1-y_i(w*x_i+b),\ i=1,2,...,N
$$
又根据第二个约束条件
$$
\zeta_i\geq 0,\ i=1,2,...,N
$$
所以，可加这两个约束条件合并为
$$
\zeta_i\geq[1-y_i(w*x_i+b)]_+,\ \ i=1,2,...,N
$$
其中，$[x]_+$表示$max(0,x)$

注意到线性SVM的优化目标中包含对$\zeta_i$的优化，对其进行极小化。而通过约束条件可以知道$\zeta_i$的下界为$[1-y_i(w*x_i+b)]_+$，极小化其下界可以等价于极小化其本身，所以可以得到无约束条件的优化式：
$$
\min_{w,b}\ \sum_{i=1}^N [1-y_i(w*x_i+b)]_++\lambda||w||^2
$$
其中，$\lambda=\frac{1}{2C}$

这就是hinge loss折页损失。

观察这个损失，可以将第一项看做主体损失，而本来的主体优化目标第二项则看做正则项，作用是防止过拟合，这是符合间隔最大化核心思想的，因为间隔最大化的目的就是防止过拟合，等同于正则。

观察第一项主体损失，$y_i(w*x_i+b)$的正负表示分类是否正确，若为负，表示分类错误，则必然有$1-y_i(w*x_i+b)>0$，即一定会产生loss；若为正，表示分类正确，但需要看值是否大于1，若小于1，表示确信度不够，仍然会产生loss，这也是加间隔（margin）的思想，这里的间隔就是1（因为优化式推导中做了归一化）。

对于这个无约束的优化问题，可以采用梯度下降法求解了。

##### 多分类SVM loss

SVM针对的是二分类问题，推广至多分类可以采用1vsN的方式。也可以类比softmax分类，先生成一个$C$维的logit向量，再做分类，$C$为类别数。

在hinge loss中，用$y_i(w*x_i+b)$来表示分类是否正确以及置信度，这个量和0的大小关系表示分类是否正确，和1的大小关系表示是否有足够的分类置信度，即是否要产生loss。

类推到多分类，设多分类任务对输入样本$x_i$输出的logit向量为$s$，样本的类别标签为$y_i$，则$s_{y_i}$为样本属于ground-truth class的分数，这个分数要足够大，要比其他类别的分数都要大，才可以使得模型正确分类。

对其他某一类j，$s_j-s_{y_i}$的值表示模型会不会把样本$x_i$错误的分成第j类，若大于0则会错误分类，为了使模型不错误分类，应该使该值比0小，同时考虑置信度或者说间隔最大化的要求，要**足够小与0**，即给这个值加上一个数（间隔，margin）后仍然小于0，对应的loss为
$$
[s_j-s_{y_i}+\triangle]_+
$$
其中，$\triangle$为间隔，是一个超参数。

考虑其他所有non ground-truth的类别，总体的SVM loss为
$$
\sum_{j\neq y_i}[s_j-s_{y_i}+\triangle]_+
$$

### 求解方法

和线性可分SVM一样，采用拉格朗日对偶性求解。

#### 1. 构造拉格朗日函数

$$
L(w,b,\zeta,\alpha,\mu)=\frac{1}{2}||w||^2+C\sum_{i=1}^N\zeta_i+\sum_{i=1}^N\alpha_i(1-\zeta_i-y_i(w*x_i+b))-\sum_{i=1}^N\mu_i\zeta_i
$$

其中，$\alpha_i\ge0$，$\mu_i\ge0$

原始的线性SVM优化问题就等价于
$$
\min_{w,b,\zeta}\max_{\alpha,\mu}L(w,b,\zeta,\alpha,\mu)
$$

#### 2. 应用拉格朗日对偶性构造对偶问题并求解

$\min_{w,b}\max_{\alpha,\mu}L(w,b,\zeta,\alpha,\mu)$的对偶问题为
$$
\max_{\alpha,\mu}\min_{w,b,\zeta}L(w,b,\zeta,\alpha,\mu)
$$
先求$\min_{w,b,\zeta}L(w,b,\zeta,\alpha,\mu)$
$$
0=\frac{\part L}{\part w}=w-\sum_{i=1}^N\alpha_iy_ix_i\\
0=\frac{\part L}{\part b}=-\sum_{i=1}^N\alpha_iy_i\\
0=\frac{\part L}{\part \zeta_i}=C-\alpha_i-\mu_i
$$
得到，
$$
w=\sum_{i=1}^N\alpha_iy_ix_i\\
\sum_{i=1}^N\alpha_iy_i=0\\
C-\alpha_i-\mu_i=0
$$
代入$L$表达式中，有
$$
L(w,b,\zeta,\alpha,\mu)=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)+\sum_{i=1}^N\alpha_i
$$
得到对偶问题：
$$
\max_{\alpha}-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)+\sum_{i=1}^N\alpha_i\\
s.t.\ \sum_{i=1}^N\alpha_iy_i=0\\
s.t \ C-\alpha_i-\mu_i=0\\
s.t.\ \alpha_i\geq0,\ i=1,2,...,N\\
s.t.\ \mu_i\geq0,\ i=1,2,...,N
$$
等价于
$$
\min_{\alpha}\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \sum_{i=1}^N\alpha_iy_i=0\\
s.t.\ 0\leq\alpha_i\leq C,\ i=1,2,...,N
$$

#### 3. 根据拉格朗日对偶性判定定理推出对偶问题的解就是原始问题的解

根据《统计学习方法》附录C.2定理，可以判定上面所求的对偶问题的解就是原始线性SVM优化问题的解。

假设对偶问题的解为$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$

#### 4. 根据对偶问题的解和KKT条件求出原始问题的解

假设原始问题的解为$w^*,b^*$，那么$\alpha^*$是对偶问题的解且$w^*,b^*$是原始问题的解的充分必要条件是$w^*,b^*,\alpha^*$满足如下的KKT条件：
$$
\frac{\part L(w^*,b^*,\zeta^*,\alpha^*,\mu^*)}{\part w}=w^*-\sum_{i=1}^N\alpha^*y_ix_i=0\\
\frac{\part L(w^*,b^*,\zeta^*,\alpha^*,\mu^*)}{\part b}=-\sum_{i=1}^N\alpha_i^*y_i=0\\
\frac{\part L(w^*,b^*,\zeta^*,\alpha^*,\mu^*)}{\part \zeta}=C-\alpha^*-\mu^*=0\\
\alpha_i^*(y_i(w^**x_i+b^*)-1+\zeta_i^*)=0,\ i=1,2,...,N\\
\mu_i^*\zeta_i^*=0\\
y_i(w^**x_i+b^*)-1+\zeta_i^*\geq0,\ i=1,2,...,N\\
\zeta_i^*\geq0,\ i=1,2,...,N\\
\alpha_i^*\geq0,\ i=1,2,...,N\\
\mu_i^*\geq0,\ i=1,2,...,N
$$
所以，得到
$$
w^*=\sum_{i=1}^N\alpha_i^*y_ix_i\\
b^*=y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i*x_j)
$$
求解过程和线性可分SVM这里是一样的。

这就是最终的线性SVM的最优化解。

可以发现，线性SVM相比线性可分SVM，转化成对偶问题后，只有约束条件$\alpha_i\leq C$这一点的区别。

最终的分类决策函数为
$$
f(x)=sign(\sum_{i=1}^N\alpha_i^*y_ix_i*x+y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i*x_j))
$$

#### 线性SVM的支持向量

$\alpha_i>0$的样本$x_i$为支持向量。

## 非线性SVM

以上两种SVM算法都是线性模型，即用一个超平面来对数据进行分类。但更多情况下，用一个超平面无法将数据集进行良好的分类，这时候需要用**超曲面**来作为分类面进行分类。即需要建立非线性模型。

非线性问题不好直接求解，解决方法是**将非线性问题转换为线性问题**。比如，一个数据集需要用椭圆作为分类线（数据为二维），分类面为：
$$
w_1(x^{(1)})^2+w_2(x^{(2)})^2+b=0
$$
如果做一个非线性变换$z=\phi(x)$，使得变换后的分类面为
$$
w_1z^{(1)}+w_2z^{(2)}+b=0
$$
这个分类面就是线性的了，即转化成了线性问题。

**经过非线性变换后的数据从原始空间（输入空间）映射到了一个新的空间（特征空间），在新的空间中用线性模型进行分类，这就是核技巧。**

（注意：核技巧是解决非线性问题建模的通用方法，非线性SVM只是其一个应用。）

### 核函数

设输入空间为$X$，特征空间为$H$，有**映射函数**：
$$
\phi(x):X\rightarrow H
$$
定义**核函数**为
$$
K(x,z)=\phi(x)*\phi(z)
$$
其中，$\phi(x)*\phi(z)$表示$\phi(x)$和$\phi(z)$的內积。

在核技巧中，一般直接定义核函数$K(x,z)$，而不显式的定义映射函数。

#### 核函数应用于SVM

注意到，线性SVM采用拉格朗日对偶性转换为对偶问题求解，对偶问题的优化目标为
$$
\min_{\alpha}\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i*x_j)-\sum_{i=1}^N\alpha_i
$$
非线性SVM需要先对输入数据用映射函数做变换，变换后的对偶问题优化目标为
$$
\min_{\alpha}\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(\phi(x_i)*\phi(x_j))-\sum_{i=1}^N\alpha_i
$$
即为
$$
\min_{\alpha}\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i
$$
最终的分类决策函数为
$$
f(x)=sign(\sum_{i=1}^N\alpha_i^*y_ix_i*x+y_j-\sum_{i=1}^N\alpha_i^*y_i(x_i*x_j))
$$
经过映射函数变换后的形式为
$$
f(x)=sign(\sum_{i=1}^N\alpha_i^*y_i\phi(x_i)*\phi(x)+y_j-\sum_{i=1}^N\alpha_i^*y_i(\phi(x_i)*\phi(x_j)))
$$
即为
$$
f(x)=sign(\sum_{i=1}^N\alpha_i^*y_iK(x_i,x)+y_j-\sum_{i=1}^N\alpha_i^*y_iK(x_i,x_j))
$$
所以可以直接用核函数来代替线性SVM优化问题中的內积。

#### 正定核*

论证$K(x,z)$能成为核函数的充要条件。

#### 常用核函数

1. 多项式核函数
   $$
   K(x,z)=(x*z+1)^p
   $$

2. 高斯核函数
   $$
   K(x,z)=\exp(-\frac{||x-z||_2^2}{2\sigma^2})
   $$

3. 字符串核函数

   有空再写

## SMO算法

求解线性SVM对偶问题的快速算法。

有空再补。

