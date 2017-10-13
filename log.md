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
*  一个Classifier多个regression models，每个regression model 对应一个位置

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