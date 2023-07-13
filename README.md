# MultimodelCommentAnalysis

## 目录结构

```
.
├── checkpoint  # 训练结果目录
├── data  # 数据目录
├── eda.py # 数据分析
├── tools
│   ├── download_img.py  # 从 Books_5.json 下载图片
│   ├── gen_samples.py  # 数据集制作
│   └── get_label.py  # 自动打标签
├── Multimodel # 多模态方案
├── SingleModel # 单模态方案
└── tools
    ├── download_img.py # 从 Books_5.json 下载图片
    ├── feature.py # 提取统计特征
    ├── gen_images.py # 图片数据集制作
    ├── gen_samples.py # 文本数据集制作
    ├── get_label.py # 自动打标签
    ├── get_user.py # 获取所有用户评论
    ├── kmean.py # 对评论聚类分析
    ├── pred.py # 无监督分类
    ├── rm_single.sh # 删除只有单个评论的用户
    ├── sub.sh # 文件分割
    ├── test.py # 
    └── user_hist.py # 绘制用户评论统计特征分布

```
