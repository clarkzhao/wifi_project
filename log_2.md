 ###Date 2017-10-13###

paper 2的要点笔记******

** **

**本文要点是基于移动手机内一般自带的传感器的数据（这里有****172 ****个属性），用多个机器学习方法对其进行训练从而解决室内定位问题。并且对每个算法的预测性能进行了比较。******

**本文中最好的算法的平均定位误差在****0.76****米左右。******

** **



关于wifi signal 强度 作为feature的算法主要分为两类；

第一类是利用 nearest neighbour（k）in signalspace；K》= 1；

第二类是更复杂的wifi signal 传播算法，计算设备距离wifi 接入点的距离。



K-NN ：平均定位误差 在两米左右，缺点太慢识别新的点；

两个定位最好的算法：

K*: 一个基于实距的有关熵的距离函数（an instance-based approach that uses an

entropy-baseddistance function）

RBFRegressor: a radial basis function network trained in a fully supervised manner;



   WaiKato Environment for Knowledge Analysis (WEKA)

（用于比较与选取不同的算法和相应的参数，可用于我们的项目中）；

更多的， 本文测试了数据点的密集度（在一个确定的室内）对定位精度的影响，（这个可以被考虑在我们的project中,overfitting or underfitting）；

 

本文中也运用了一个分步分类的方法提高算法识别的速度，如k-NN，可将室内的点总共分为几部分，对于一个新数据点，先用如随机森林方法训练识别属于哪个部分；再用KNN 找到属于该点的分类。

 

** **

** **

 ###Date 2017 -10-15###

** paper 3  notes important !!! **

Stacked Autoencoder 能有效减少特征空间的维度通过识别关键的信息；

这里可以很好的运用在wifi project 中由于在一个数据点中许多wifi 信号未被体现；



利用wifi信号定位的方法大概可以分两种，wifi signal wave 受太多因素影响在室内，故使用wifi fingerprint更精准一些；

 

在室内定位问题中，假使已经识别出楼层，最基础的方法是使用KNN or Weight KNN 去定位；假使数据量很大，花费的训练和认为调参时间会很多；

 

本文中state of the art 的结构是一个DNN 分辨器连接一个自动编码器；

首先 输入encoder的数据结构（529个维度）包括室内所有的wifi scan 信号（520个在本文中）；其他九个维度是基础信息，如latitude 和attitude and user ID 等；

由于每个数据点只有一部分wifi信号可以收集的到；按此来说每个数据存在大量的维度缺少，因此使用stacked-Autoencoder变的十分合适去减少原始数据的representation；有效的提取关键信息；

然后encoder的output 输入一个分类器，中途通过几层网络和处理（dropout）；最后结果是一层softmax，神经元数目等于所有大楼及层数的总和（在我们的project，相应的是所有商铺的总和）；

文中是用 cross entropy error 和adam 加速器；



Random 划分 training和validation，test data；

 

**如上文所述，每个数据点中缺失 wifi信号的 representing**

在机器学习方法中，主要的问题是如何表达大量确实的AP点（等同于feature这里）；

本文定义强度从0到-104 dBm逐渐减少，而缺失的测量定义为100 或-110（实验发现定义为同符号的-110 具有更好的效果）；

 

调整wifi 测量值使其的平均值和方差分别为0和1，更多的还区分了独立缩放（scaled）和共同缩放的效果；

最好的结构及其结果如下：

Stacked autoencoder 256-128-64 classifier 128 128（但因为数据点数目及维度的不同，我们得自己设置相应的网络和神经元数目），the scaled of lack measurements joint 有更好的效果比independent.

#### ~大概步骤~:###

1，数据预处理，根据mall id 将 数据分组进行训练（可能性价比更高，负样本对结果影响不大相对于正样本）

2，先训练一个stacked autoencoder，然后将encoder 的结果连接一个full connected的 neural network 进行训练。





 

 

 