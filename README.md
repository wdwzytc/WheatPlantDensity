# WheatPlantDensity 小麦植株密度

Estimate plant density for wheat-like plants at early stages, using leaf tip detection and analysis of the dynamic.

基于叶尖检测与动态分析，实现小麦等作物苗期植株密度估算。

# Leaf Tip Detection Models for Wheat 小麦叶尖检测模型

This repository provides an **early / core release** of leaf tip detection models developed for wheat plant phenotyping, with a focus on plant density estimation and dynamic field phenotyping.

本仓库提供用于小麦植株表型分析的**早期核心版**叶尖检测模型，重点面向田间植株密度估算与动态表型分析。

The models are designed to detect wheat leaf tips under field conditions and are intended for research and academic use.

模型面向田间环境设计，用于小麦叶尖检测，**仅供科研与学术用途**。

---

# v0.1

...

**Last update:** 2026-03-04
**最后更新：** 2026-03-04

**Maintainer:** Tiancheng
**维护者：** Tiancheng

---

## 📦 Released Content (Core Version) 发布内容（核心版）

### 1. YOLOv8-based Leaf Tip Detection Model 基于 YOLOv8 的叶尖检测模型

- Model: YOLOv8 (https://github.com/ultralytics/ultralytics/tree/main)
- 模型：YOLOv8 (https://github.com/ultralytics/ultralytics/tree/main)

- Task: Wheat leaf tip detection
- 任务：小麦叶尖检测

- Weights:
  - 模型权重:
  - https://huggingface.co/TC0/YOLOv8_for_wheat_leaf_tip_detection

> Note: This model was trained using the Ultralytics YOLOv8 framework.
> 说明：本模型基于 Ultralytics YOLOv8 框架训练。

---

### 2. P2PNet-based Leaf Tip Detection Model 基于 P2PNet 的叶尖检测模型

- Model: P2PNet (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
- 模型：P2PNet (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)

- Task: Wheat leaf tip detection (point-based)
- 任务：小麦叶尖检测（基于点目标）

- Weights:
  - 模型权重:
  - https://huggingface.co/TC0/P2PNet_for_leaf_tip_detection

---

## 📜 License 许可证

- **YOLOv8-based model**: 基于 YOLOv8 的模型：
  Released under **AGPL-3.0**, as it is trained using the Ultralytics YOLOv8 framework.
  采用 **AGPL-3.0** 协议开源，因模型基于 Ultralytics YOLOv8 框架训练。

- **P2PNet-based model**: 基于 P2PNet 的模型：
  License will follow the original P2PNet implementation and will be clarified in the full release.
  许可证遵循 P2PNet 原始开源协议，完整版本中将进一步明确。

Please ensure compliance with the corresponding licenses when using the models.
使用模型时请遵守对应开源许可证条款。

---

## 📚 Paper 论文

https://doi.org/10.1016/j.compag.2025.111053
