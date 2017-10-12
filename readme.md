[TOC]

# 项目简介

## [商场中精确定位用户所在店铺](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.35ed5dcatQsSbv&raceId=231620)

**训练集**：2017年8月份数据进行店铺、用户、WIFI等各个维度的数据

**测试集**：2017年9月份的数据

**目标**：在测试集中根据当时用户所处的位置和WIFI等环境信息，通过您的算法或模型准确的判断出他当前所在的店铺。

## 时间

**初赛（10月10日-11月19日）**

A榜评测时间：10.16-11.16；B榜评测时间：11.17-11.19

**复赛（11月21日—12月11日）**

A榜评测时间：11月21日-12月4日，B榜评测时间：12月5日-12月11日

## [数据集](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100069.5678.2.3f69082f9OuKsK&raceId=231620)

### Table 1、店铺和商场信息表

| Field       | Type   | Description | Note                      |
| ----------- | ------ | ----------- | ------------------------- |
| shop_id     | String | 店铺ID        | 已脱敏                       |
| category_id | String | 店铺类型ID      | 共40种左右类型，已脱敏              |
| longitude   | Double | 店铺位置-经度     | 已脱敏，但相对距离依然可信             |
| latitude    | Double | 店铺位置-纬度     | 已脱敏，但相对距离依然可信             |
| price       | Bigint | 人均消费指数      | 从人均消费额脱敏而来，越高表示本店的人均消费额越高 |
| mall_id     | String | 店铺所在商场ID    | 已脱敏                       |



### Table 2、用户在店铺内交易表

| Field      | Type   | Description                              | Note                                     |
| ---------- | ------ | ---------------------------------------- | ---------------------------------------- |
| user_id    | String | 用户ID                                     | 已脱敏                                      |
| shop_id    | String | 用户所在店铺ID                                 | 已脱敏。这里是用户当前所在的店铺，可以做训练的正样本。（此商场的所有其他店铺可以作为训练的负样本） |
| time_stamp | String | 行为时间戳                                    | 粒度为10分钟级别。例如：2017-08-06 21:20            |
| longitude  | Double | 行为发生时位置-精度                               | 已脱敏，但相对距离依然可信                            |
| latitude   | Double | 行为发生时位置-纬度                               | 已脱敏，但相对距离依然可信                            |
| wifi_infos | String | 行为发生时Wifi环境，包括bssid（wifi唯一识别码），signal（强度），flag（是否连接） | 例子：b_6396480\|-67\|false;b_41124514\|-86\|false;b_28723327\|-90\|false;解释：以分号隔开的WIFI列表。对每个WIFI数据包含三项：b_6396480是脱敏后的bssid，-67是signal强度，数值越大表示信号越强，false表示当前用户没有连接此WIFI（true表示连接）。 |

### Table 3、评测集

测试数据A榜和B榜格式相同，只是选取的时间不同，A榜数据是9月份第一周数据，B榜数据是9月份第二周数据。

| Field      | Type   | Description                              | Note                          |
| ---------- | ------ | ---------------------------------------- | ----------------------------- |
| row_id     | String | 测试数据ID                                   |                               |
| user_id    | String | 用户ID                                     | 已脱敏，并和训练数据保持一致                |
| mall_id    | String | 商场ID                                     | 已脱敏，并和训练数据保持一致                |
| time_stamp | String | 行为时间戳                                    | 粒度为10分钟级别。例如：2017-08-06 21:20 |
| longitude  | Double | 行为发生时位置-精度                               | 已脱敏，但相对距离依然可信                 |
| latitude   | Double | 行为发生时位置-纬度                               | 已脱敏，但相对距离依然可信                 |
| wifi_infos | String | 行为发生时Wifi环境，包括bssid（wifi唯一识别码），signal（强度），flag（是否连接） | 格式和训练数据中wifi_infos格式相同        |

### Table 4、选手需要提交的结果，统一命名为：result

文件中包含字段如下：row_id，shop_id，第一列 row_id 与第二列 shop_id 请使用英文逗号分隔并提交 csv（Comma-Separated Values）格式的文件。空的或者无效的row_id都会被自动忽略掉，空的或者无效的shop_id也会被忽略。

| Field   | Type   | Description | Note            |
| ------- | ------ | ----------- | --------------- |
| row_id  | String | 测试数据ID      | 下载的测试文件中的row_id |
| shop_id | String | 店铺ID        | 算法检测的结果         |

答案样本

| row_id | shop_id |
| ------ | ------- |
| 1      | xx      |
| 4      | xx      |
| 7      | xx      |
| 10     | xx      |



# 有用的链接 

### 手机wifi定位相关

1. [smartphone-based offline indoor location](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5375843/)


### 计算机相关

1. [团队使用github](http://www.cnblogs.com/zhangchenliang/p/3950778.html)

### 数据挖掘相关

1. [爱丁堡大学数据挖掘课程代码库](https://github.com/agamemnonc/dme)