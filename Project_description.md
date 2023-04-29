# 项目文件说明

## 2023.4.22——version 0.1.2

- 新增：

  1、新增Unet 3+

  2、新增decoder文件夹，统一Unet、Unet 3+的decoder模块定义。

- 修改：

  1、修改所有模型__init__函数，删除了之前多余的成员变量定义，减小了显存占用量

  2、修改了./blocks/conv中的一些模块定义，减少显存占用



## 2023.4.22——version 0.1.1

- 新增：

  1、新增AttUnet

- 修复：

  1、修改resnet，将其中bottle_neck结构的expansion设置改为2，并统一了每个层的输出通道与原版unet_backbone每层输出通道数一致。



## 2023.4.18——version 0.1

### 文件夹结构

- data（数据文件夹）

  - MICCAI_pre_test_data（报名前测验数据集，训练集：92884；测试集（无mask）：25027）

    - train（原数据采样3/4）

      - image（67259张，542个patient）

        - patient 1

        - patient 2

          .......

      - mask（23625张，180个patient）

    - validate（原数据采样1/4）

    - TestImage（100个病人）

- general_seg_nets（项目文件夹）
  - backbone（主干网络，粗体为已经写好的）
    - efficient_net.py
    - mae.py
    - mobilenet_v3.py
    - resnest.py
    - **resnet.py**
    - **resnext.py**
    - swin.py
    - **unet_backbone.py**
    - **VGG.py**
    - vit.py
  - checkpoints（存放训练权重）
  - configs*（暂时还没用到）*
  - dataset
    - playground_dataset.py（加载预测试数据集使用的数据加载类）
  - inits
    - init.py（包括一些层初始化和网络初始化函数）
  - models（分割网络）
    - Unet.py（Unet）
  - train.py（训练脚本）
  - util.py（一些杂七杂八方便使用的函数）
  - debug.py（草稿本罢咧~）

​	