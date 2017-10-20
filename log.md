# 项目进展报告

## 2017-10-13 00:05:48 by SY

训练数据保存在本地上，不然上传到云上太慢了。

我在 **.gitignore**文件上写了让git不记录**./data/**下的数据变化。

三个训练文件放在了**/wifi_project/data/**下，都把中文名部分删除，方便读取：

**evaluation_public.csv**

**ccf_first_round_shop_info.csv**

**ccf_first_round_user_shop_behavior.csv**



## 2017-10-13 12:38:48 by SY

### paper笔记

论文题目

***WiFi Localization For Mobile Robots based on Random Forest and GPLVM by R. Elbasiony & W. Gomaa.***

#### 训练集：

*  一个机器人上探测的所有WiFi和WiFi信号的强度，这是我们可以借鉴的。
*  这些WiFi强度是在一条预先设定好的路线上，让机器人边走边测得，这和我们仅仅在顾客交易的时候读取到的WiFi强度是不一样的。

#### 测试集

*  WiFi信号强度，但是机器人走的路线是未知的。

#### 方法

1. 训练的时候

*  Random Forest Classifier
   *  输入
      *  一组WiFi 信号强度
   *  输出
      *  预测对应机器人的地理位置
*  Random Forest Regression
   *  输入
      *  xy 坐标，也就是地理位置
   *  输出
      *  WiFi 信号强度
*  一个Classifier，多个regression models，每个regression model 对应一个位置

2. 预测的时候

*  WiFi classifier
   *  就是用训练好的RF Classifier，输入Wifi 信号强度，输出位置

***但是将位置预测问题表示成分类问题是有limitation的***

*  每个训练点是有间隔的，就算是最完美的分类器也会有很大的误差。


*  文章中运用了 random location generator 的方法。
   *  这个random generator用二元高斯（因为位置是用xy坐标表示的）采样了很多个点
   *  高斯分布的mean是分类器输出的xy坐标的均值
   *  variance是从测试集来的
   *  也就是说，random location generator扩大了输出的点范围，而不是直接用classifier输出的点


*  WiFi Regressor
   *  是Random Forest regressor输出WiFi信号强度，输入xy坐标。
   *  这些输出的xy坐标是刚刚很多的random location genrator采样出来的位置
   *  这样我们就有了这些采样的的位置和相应的用WiFi Regressor预测出来的WiFi信号强度了。
*  WiFi Comparator and Selector
   *  比较两种WiFI信号强度的相似度，一种是测试集的输入，另一种是Random Forest Regression输出的WiFi信号强度。
   *  输出的位置是WiFI信号相似度最大的位置

#### 思考

1. 我们有每个店铺的位置信息，只要我们的模型预测出了位置信息（longtidue，latitude），就可以选择最接近的店铺作为我们的分类结果。但是在测试集中，位置信息已经给我们了，难道说我们不用训练模型了，只需要选择位置最近的店铺作为分类结果么？然后我发现了，其实在一个商场的店铺的地理位置是非常非常接近的，位置信息的差别在千分位乃，至万分位。所以仅仅凭借位置信息，我们无法预测出商店的。
2. 这时候我们就需要WiFi信息了。在训练集，用户在店交易信息中，我们有WiFi强度信号。但是我们除了经纬度，没有更精确的位置坐标了，有些店铺可能在楼上楼下，尽管店铺不同，但是经纬度是几乎一样的，当我们确定了商场以后，经纬度似乎就没有用了。在这篇论文中，每个训练样本的位置是很精确的，而且WiFi点也是固定的几个。然而我们的数据集中，同一个商场的WiFi是不是一样，有没有随机的outlier，比如别人开的热点什么的，都有待确认。
3. ***所以目前，我觉得我可以先进行数据清洗，讲每个商场的固定的WiFi点洗出来，作为我们的Feature。***
4. 然后可以试试用random Forest算法分类，回归什么的。





## 2017-10-15 09:13:02 by SY

### paper笔记

#### 题目

***Low-eﬀort place recognition with WiFi ﬁngerprints using deep learning***

#### 主要贡献

*  使用了stacked autoencoder（SAE） 的方法将WiFi强度信号特征降维，这提高了神经网络在测试集上的泛化性能。
*  作者尝试了不同的神经网络大小，选择了最优的。***但是我们的特征空间不一样大，所有我们得自己找到更适合我们的神经网络超参数。***

#### 数据集

UJIIndoorLoc dataset that contains WiFi measurements used during EvAAL competition at IPIN 2015

总共21048 WiFi scans that are divided into 19937 training and 1111 validation samples。

这些Scans是取自西班牙某大学的建筑物里，涵盖了110000平方米的距离。***但是我们的数据集并不在一个区域内，所以可以尝试将数据集分成97个子数据集，每个子数据集对应一个mall。亦或者用one-hot的方法，将mall_id特征变成97个关于mall_id的特征，让模型自己去学习，mall_id和shop_id的相关性***

总共529个特征，其中520个特征是所有可被发现的wifi信号强度。强度范围在-104dBm（最弱）到0dBm（最强）

#### 目标

是将结果分类成每个建筑物的楼层，如果一共有N个建筑，每个建筑i，有M_i层，那么输出神经元的个数就是：
$$
\sum_{i=1}^NM_i
$$
***我们的目标函数是每个商场的商店，这要求模型的精度会比只分类到楼层数更高，这可能会导致我们模型的效果到不如这个paper上描述的。***

#### 数据预处理

wifi无法被发现时，作者把强度设置成了0，后面的实验发现，设置成-110在测试集上效果更好。

wifi信号强度被标准化了，标准化的方法有两种，一种是对每个wifi单独标准化，即使数据零均值，单位方差。另一种是基于所有wifi强度，**一起标准化**。

***结果发现一起标准化，和把信号强度缺失值设置成-110更合理，这是我们可以借鉴的。***

剩下9个特征包括了longitude and latitude of measurement, ﬂoor number, building ID, space ID, relative position, user ID, phone ID and the timestamp of the measurement。

#### 训练方法

先训练一个Autoencoder，Autoencoder分为encoder，decoder部分，训练完Auto恩code以后，舍弃decoder部分，在encoder后接一个全连接神经网络。图见paper。

Loss：categorial cross entropy error

optimizer：Adam 

框架：Keras（TensorFlow）, Scikit-learn

### 开发进度

1. 根据店铺和商场信息表，用dictionary建立关于`shop_id`到`mall_id`的映射，取名叫`shop2mall`
2. 根据训练集`user_train`中`shop_id`的信息，增加一列特征，增加后的新训练集叫`user_train_mall_added`



## 2017-10-17 11:15:13 by SY

### 开发进度

1. 根据mall_id将训练集分割成97组
2. 建立了预处理数据的三个方程
   1. `filter_bssid`用来筛选掉无效的wifi
   2. `expand_wifi_feature`用来拓展特征空间
   3. `standardize`用来标准化经纬度，wifi信号强度




## 2017-10-18 23:15:05 by SY

### 开发进度

所有代码基于python 2.7

### Preprocess.py

1. Option 1可以让你预处理根据shop_id分割的小数据集，可以更改第`210`行代码的for循环，少量的，预处理一部分shop_mall。比如变成`for df in train_sets[:1]:`就可以只预处理第一组子数据集。若不改，就是输出所以子数据集，会跑的比较长。
2. Option 2 是预处理整体的数据集，目前数据集太大，内存爆炸，电脑不会跑出结果。可行的解决方案是分割整体数据集，一小份一小份输出。目前不建议尝试。
3. 更改KEEP_PER，可以改变空间的大小，具体请看注释。

#### model_by_sy.py

1. 我的深度学习训练代码。

#### Data_providers.py

    1.  我将我们深度学习课程的data_providers稍作改动，创建了类 WIFIDataProviders

    2.  这个类可以比较方便的读取数据，数据目前存储比较笨，要三份，一份整体数据，一份train，一份valid，主要原因是，我需要one-hot所有的shop_id，所以必须要一份整体的数据。这三份数据可以用preprocess.py文件输出。

    3.  具体使用的时候

      *  `train_data = data_providers.WIFIDataProvider(mall_id, 'train', batch_size=64)`
      *  这里，mall_id可以是比如‘m_625’，‘train’指定了是训练集，变成‘valid’就是验证集了，batch_size也可以选
      *  `train_data.next()`可以输出下一个batch的inputs和targets，targets自动one-hot
      *  也可以用这种pythonic的代码`for input_batch, target_batch in train_data：`循环遍历整个训练集。具体见`model_by_sy.py`



### 明日计划

1. 分割evaluation数据集（根据mall_id）。
2. 用最简单的mlp模型训练97次，保存97个模型。
3. 用97个模型，分别预测分割好的evaluation数据集，整合预测结果，上传测评。

### Feedback to xiaohao:

1. 那就照你说的做吧，研究一下VAE，而我就不和你看重复的了，我后天如何用集成学习（Boost等等）训练多个小模型，最后集合成一个强模型（这个方法在数据比赛中非常常用），以及Cross validation等等的方法。



## 2017-10-19 14:56:34 by SY

### 开发进度

1. 总体上完成了昨日的计划，等待18:00的排名，看看有没有bug
2. 增加了Option 3 在`preprocess.py`中，用来预处理evalution data，注意evalution data和原始训练数据略有不同，比如shop_id的位置，等等。为了今后处理方便统一了格式。以及要保留的bssids要从已经生成的valid数据里读取。
3. 在`model_by_sy.py`中，增加了可以保存模型的功能，每个模型训练20个epoch，只有valid_error最小的模型会被留下。
4. 在`eval.py`中，我们首先要改动一个变量`timestamp_from_training`就是刚刚训练时生成的时间戳，告诉电脑哪个文件夹找。



### 未来计划

1. 增加模型复杂度，等待vae效果，但是受限于97个子样本的小容量，过于复杂的模型可能无法收敛。
2. 研究集成学习（Boost等等）



# 2017-10-19 22:59:27 by SY

## 开发进度

1. 修复了在evaluation中shuffle数据导致row_id不匹配的bug
2. 修改了DNN模型，注意训练模型和验证模型的tensor计算图必须完全一致（这部分代码可以放一起，增加代码的鲁棒性，以后再改），不然会有bug。
3. 在原始文件夹增加了一个`shop_in_mall.npy`文件，是一个python字典，key是`mall_id`，value是一个包含所有`shop_id`的list，这数据是从原始商店数据中提取的，为了方便的给其他代码引用，我就索性放在src里了。代码也进行了相应的简化。



# 2017年10月20日22:15:37

## 开发进度

1. 将原文件中的constants集合在`constant.py`上
2. 修改了preprocess，减少了以一半的存储量
3. 验证了当预处理中，**常数keep_per增加到1e-3时**，features总数量大幅减少约50%，模型效果减少约10%
4. 等待hidden_layer_2=256是模型test结果
5. 修复了preprocess中的bug，因为一次scan中，可能存在多个相同的bssids，原来的方式是保存最后一个，该相同bssid的强度，目前方法是如果一次scan，扫描到多个相同的bssids，选取强度大的那个。

