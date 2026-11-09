# 黄斑中央凹定位

这个项目使用带全图上下文分支的U-Net，从眼底图像预测黄斑中央凹的原图坐标。训练和预测在Kaggle Notebook中完成。

## 代码文件

| 文件 | 用途 |
| --- | --- |
| `cell_01_config_and_data.py` | 导入依赖，设置训练参数，读取图片和标注并划分验证集 |
| `cell_02_model_and_loss.py` | 定义图像增强、定位模型、损失函数和推理方法 |
| `cell_03_train_and_predict.py` | 使用3个随机种子进行5折训练，汇总各模型的预测 |
| `cell_04_export_submission.py` | 汇总验证结果，按比赛样例导出提交文件 |

## 数据集

在Kaggle Notebook中添加黄斑中央凹定位比赛的数据。数据包括 `train/` 下的80张训练图像和 `test/` 下的20张测试图像。`train_location/` 保存训练图像的XML标注框，`fovea_localization_train_GT.csv` 保存中央凹坐标，`sample_submission.csv` 提供提交格式。

## 在Kaggle上运行

1. 添加比赛数据并开启GPU。代码会自动查找数据；如果找不到，在 `cell_01_config_and_data.py` 中将 `DATA_ROOT` 设为包含训练标注CSV、`train/` 和 `test/` 的目录。
2. 确认环境包含PyTorch、NumPy、pandas和OpenCV。模型从头训练，无需下载预训练权重。
3. 将4个 `.py` 文件的内容分别放到同一个Notebook的4个代码单元格中，按 `01` 到 `04` 的顺序运行。后面的单元格会用到前面定义的变量和函数。
4. 运行完成后，从 `/kaggle/working/submission_competition_only.csv` 下载提交文件。每张测试图的X和Y坐标按样例拆成两行。
