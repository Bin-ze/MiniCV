# MiniCV
## 工具箱介绍：
使用注册器实现神经网络注册，使用Trainer实现训练流程，用于图像分类任务
## 数据准备：标准分类数据集格式
        |- dataset_name
            |- train                 # 自定义数据集的训练数据
                    |- class1
                        |-img1.jpg
                        |-img2.jpg
                        ...
                    |- class2  
                         |-img1.jpg
                         |-img2.jpg
                         ...
                    ...
            |- val                  # 自定义数据集的验证数据
                    |- class1
                        |-img1.jpg
                        |-img2.jpg
                        ...
                    |- class2  
                         |-img1.jpg
                         |-img2.jpg
                         ...
                    ...

## 使用流程
1. 参考cla_config编辑自己的配置
2. 使用train.py发起训练
3. 使用val.py进行分类结果测试
4. 使用test.py进行推理，支持批量推理

## 参考
这里是我深度学习入门的地方：
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
在此表示感谢！
