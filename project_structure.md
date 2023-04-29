## 文件夹结构——version 0.1.2

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
  - blocks
    - att_block.py
    - conv.py
    - segment_head.py*（暂时还没用到）*
    - upsample.py
  - checkpoints（存放训练权重）
  - configs*（暂时还没用到）*
  - dataset
    - playground_dataset.py（加载预测试数据集使用的数据加载类）
  - decoder
    - AttUnet_decoder.py
    - Unet_decoder.py
  - inits
    - init.py（包括一些层初始化和网络初始化函数）
  - loss（暂未补充）
  - models（分割网络）
    - Unet.py（Unet）
    - AttUnet.py（AttUnet）
    - Unet_3plus.py（Unet 3+）
  - train.py（训练脚本）
  - util.py（一些杂七杂八方便使用的函数）
  - debug.py（草稿本罢咧~）

​	