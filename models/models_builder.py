# ========================================================
# 文件名: models/models_builder.py
# 作用: 统一管理实验中所有对比大模型的构建函数 (DenseNet, ResNet等)
# 说明: 原文件位于本地工作副本 E:\imageDesigning\TermiteClassifier\models_builder.py，
#       此前被 train_termite_cls.py 引用但未提交到仓库，现随 models/ 包一并发布。
# ========================================================
import torch
import torch.nn as nn
import torchvision.models as models


def build_densenet_model(model_name="densenet121", num_classes=8, pretrained=True):
    """
    高效构建 DenseNet 并在白蚁分类数据集上运行
    """
    if model_name.lower() == "densenet121":
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
            print(">> [模块化调用] 成功加载官方 DenseNet121 预训练骨干")
        except AttributeError:
            model = models.densenet121(pretrained=pretrained)
            print(">> [模块化调用] 成功加载老版本 DenseNet121")

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    elif model_name.lower() == "densenet161":
        try:
            weights = models.DenseNet161_Weights.DEFAULT if pretrained else None
            model = models.densenet161(weights=weights)
            print(">> [模块化调用] 成功加载官方 DenseNet161 预训练骨干")
        except AttributeError:
            model = models.densenet161(pretrained=pretrained)
            print(">> [模块化调用] 成功加载老版本 DenseNet161")

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"暂不支持的模型名字: {model_name}")

    return model