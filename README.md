# WheatPlantDensity
Estimate plant density for wheat-like plants at early stages, using leaf tip detection and analysis of the dynamic.

# Leaf Tip Detection Models for Wheat

This repository provides an **early / core release** of leaf tip detection models
developed for wheat plant phenotyping, with a focus on plant density estimation
and dynamic field phenotyping.

The models are designed to detect wheat leaf tips under field conditions and are
intended for research and academic use.

---

## ⚠️ Current Status (Early Release)

The **full codebase and complete models are still under active organization and cleanup**.
Due to limited time, only the **core components (trained weights and minimal supporting materials)** are released at this stage.

> The complete and fully documented version (including training scripts, evaluation pipeline,
> and extended experiments) will be released in a future update.

**Last update:** 2026-02-04  
**Maintainer:** Tiancheng

---

## 📦 Released Content (Core Version)

### 1. YOLOv8-based Leaf Tip Detection Model

- Model: YOLOv8
- Task: Wheat leaf tip detection
- Weights:
  - https://huggingface.co/TC0/YOLOv8_for_wheat_leaf_tip_detection

> Note: This model was trained using the Ultralytics YOLOv8 framework.

---

### 2. P2PNet-based Leaf Tip Detection Model

- Model: P2PNet
- Task: Wheat leaf tip detection (point-based)
- Weights:
  - https://huggingface.co/TC0/P2PNet_for_leaf_tip_detection

---

## 🧭 Planned Release

The following components are planned for future release:

- Full training and inference code
- Data preprocessing and annotation pipeline
- Model evaluation scripts
- Detailed documentation and usage examples
- Extended experiments and ablation studies

---

## 📜 License

- **YOLOv8-based model**:  
  Released under **AGPL-3.0**, as it is trained using the Ultralytics YOLOv8 framework.

- **P2PNet-based model**:  
  License will follow the original P2PNet implementation and will be clarified in the full release.

Please ensure compliance with the corresponding licenses when using the models.

---

## 📚 Citation

If you use this work in your research, please cite it as follows:

```bibtex
@misc{tiancheng_leaf_tip_2026,
  title  = {Leaf Tip Detection Models for Wheat},
  author = {Tiancheng},
  year   = {2026},
  note   = {Early release}
}
