# 项目进展报告

2017-10-13 00:05:48 by SY

训练数据保存在本地上，不然上传到云上太慢了。

我在 **.gitignore**文件上写了让git不记录**./data/**下的数据变化。

三个训练文件放在了**/wifi_project/data/input/**下，都把中文名部分删除，方便读取：

**evaluation_public.csv**

**ccf_first_round_shop_info.csv**

**ccf_first_round_user_shop_behavior.csv**

