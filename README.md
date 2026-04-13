# WheatPlantDensity 小麦植株密度

Estimate plant density for wheat-like plants at early stages, using leaf tip detection and analysis of the dynamic.

基于叶尖检测与动态分析，实现小麦等作物苗期植株密度估算。

---

# How to use 使用说明

1. Download this project from GitHub.
   在 GitHub 上下载本项目。

2. Download the `data` folder and place it in the root directory of this project.
   下载 `data` 文件夹并放在本项目根目录。
   - Zenodo: https://zenodo.org/records/19492894
   - HuggingFace: https://huggingface.co/TC0/WheatPlantDensityData

3. Create and activate the Python environment with Conda:
   使用 Conda 创建并激活 Python 环境：```conda create -n wheat_plant_density_env python=3.10 numpy=1.26 pandas scipy matplotlib scikit-learn tqdm openpyxl pillow albumentations=1.4.10 ultralytics=8.2.69 pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 pytorch-cuda=11.8 -c nvidia -c pytorch -c conda-forge -c defaults```

---

# v0.1 更新时间

**Last update:** 2026-04-10
**最后更新：** 2026-04-10

**Maintainer:** YANG Tiancheng
**维护者：** 杨天成

---

## 📦 Released Content 发布内容

### 1. YOLOv8-based Leaf Tip Detection Model 基于 YOLOv8 的叶尖检测模型

- Model: YOLOv8 (https://github.com/ultralytics/ultralytics/tree/main)
- 模型：YOLOv8 (https://github.com/ultralytics/ultralytics/tree/main)

- Task: Wheat leaf tip detection
- 任务：小麦叶尖检测

- 代码位置：/step1_detection/yolov8/yolov8.py
- Code Location：/step1_detection/yolov8/yolov8.py

---

### 2. P2PNet-based Leaf Tip Detection Model 基于 P2PNet 的叶尖检测模型

- Model: P2PNet (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
- 模型：P2PNet (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)

- Task: Wheat leaf tip detection (point-based)
- 任务：小麦叶尖检测（基于点目标）

- Code Location：/step1_detection/p2pnet/p2pnet.py
- 代码位置：/step1_detection/p2pnet/p2pnet.py

---

### 3. Leaf Tip Dynamic Estimation Model 基于叶尖动态的植株密度估测模型

- Method: 
- 方法：叶片动态模型 + 查找表反演

- Task: Wheat leaf tip detection (point-based)
- 任务：小麦叶尖检测（基于点目标）

- Code Location：/step2_dynamic_estimation/main.py
- 代码位置：/step2_dynamic_estimation/main.py

---

## 📜 License 许可证

- **YOLOv8-based model**: 基于 YOLOv8 的模型：
  Released under **AGPL-3.0**, as it is trained using the Ultralytics YOLOv8 framework.
  采用 **AGPL-3.0** 协议开源，因模型基于 Ultralytics YOLOv8 框架训练。

- **P2PNet-based model**: 基于 P2PNet 的模型：
  License will follow the original P2PNet implementation, and is only for research use.
  许可证遵循 P2PNet 原始开源协议，仅供学术使用。

Please ensure compliance with the corresponding licenses when using the models.
使用模型时请遵守对应开源许可证条款。

---

## 📚 Paper 论文

https://doi.org/10.1016/j.compag.2025.111053
