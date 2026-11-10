深度学习目标检测算法Yolov8训练 木材计数数据集 通过训练出的模型 建立基于深度学习的木材计数检测识别 训练和推理
@[toc]

木材圆木木材计数数据集，3721张，yolo和voc标注
1类，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8fa648ac22c46e2805f251383ca6be2.png)

标注数量：
Logs: 590469
image num: 3721
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5fec9fa132642859f8c06a5921e0b03.png)
1
*木材圆木计数检测数据集**（1类，59万+标注框，3721张图像），使用 **YOLOv8** 进行训练和推理，
代码实现：仅供参考，

---

## 🧾 一、项目结构要求

确保你的数据格式如下：

```
logs_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

---

## 📁 二、数据准备

### 1. 格式转换（VOC转YOLO）

如果你的数据是 VOC XML 格式，需要先转换为 YOLO 标注格式（txt 文件）。

```python
import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(voc_path, yolo_path, img_width, img_height):
    tree = ET.parse(voc_path)
    root = tree.getroot()

    with open(yolo_path, 'w') as f:
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls != "Logs":
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            xc = (b[0] + b[1]) / (2 * img_width)
            yc = (b[2] + b[3]) / (2 * img_height)
            w = (b[1] - b[0]) / img_width
            h = (b[3] - b[2]) / img_height
            f.write(f"0 {xc} {yc} {w} {h}\n")
```

### 2. 数据划分（train/val/test）

```python
import os
import random
import shutil

image_dir = "path/to/images"
label_dir = "path/to/labels"

images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(images)

# 创建目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(f'logs_dataset/images/{split}', exist_ok=True)
    os.makedirs(f'logs_dataset/labels/{split}', exist_ok=True)

# 划分比例：70% train, 15% val, 15% test
train_split = int(0.7 * len(images))
val_split = int(0.85 * len(images))

for i, img_file in enumerate(images):
    label_file = img_file.replace('.jpg', '.txt')
    src_img = os.path.join(image_dir, img_file)
    src_lbl = os.path.join(label_dir, label_file)

    if i < train_split:
        dst_img = os.path.join("logs_dataset/images/train", img_file)
        dst_lbl = os.path.join("logs_dataset/labels/train", label_file)
    elif i < val_split:
        dst_img = os.path.join("logs_dataset/images/val", img_file)
        dst_lbl = os.path.join("logs_dataset/labels/val", label_file)
    else:
        dst_img = os.path.join("logs_dataset/images/test", img_file)
        dst_lbl = os.path.join("logs_dataset/labels/test", label_file)

    shutil.copy(src_img, dst_img)
    shutil.copy(src_lbl, dst_lbl)
```

---

## 📦 三、环境搭建

安装 Ultralytics YOLOv8：

```bash
pip install ultralytics
```

---

## 📄 四、配置文件 `data.yaml`

创建 `data.yaml` 文件内容如下：

```yaml
train: logs_dataset/images/train
val: logs_dataset/images/val
test: logs_dataset/images/test

nc: 1
names: ['Logs']
```

---

## 🔧 五、模型训练（YOLOv8）

### 使用预训练模型进行微调（推荐）

```python
from ultralytics import YOLO

# 加载预训练模型（例如 yolov8s.pt）
model = YOLO("yolov8s.pt")

# 开始训练
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="logs_detector",
    save_period=5,
    patience=20,
    verbose=True
)
```

> ✅ 建议使用 GPU 训练，可设置 `device='cuda:0'` 或多卡训练。

---

## 🧪 六、模型推理（单图 & 批量）

### 单图推理：

```python
model = YOLO("runs/detect/logs_detector/weights/best.pt")
img_path = "path/to/test_image.jpg"
results = model.predict(source=img_path, show=False, save=True)
```

### 批量推理整个文件夹：

```python
folder_path = "logs_dataset/images/test"
results = model.predict(source=folder_path, save=True)
```

---

## 📊 七、性能评估

在验证集上评估 mAP 等指标：

```python
metrics = model.val()
print("mAP@0.5-0.95:", metrics.box.map)
print("mAP@0.5:", metrics.box.map50)
print("Recall:", metrics.box.recall.mean())
```

---

## 🧠 八、优化建议（提升精度）

1. **增加数据增强**：
   ```python
   results = model.train(
       ...
       hsv_h=0.015,
       hsv_s=0.7,
       hsv_v=0.4,
       degrees=10,
       translate=0.1,
       scale=0.2,
       shear=0.2,
       perspective=0.0001
   )
   ```

2. **调整锚框尺寸**（适合大目标）：
   ```bash
   # 在 train.py 中设置自定义 anchor
   anchors = [[10,13, 16,30, 33,23],  # layer 0
              [30,61, 62,45, 59,119], # layer 1
              [116,90, 156,198, 373,326]] # layer 2
   ```

3. **使用 mosaic 和 mixup 增强**：
   ```python
   results = model.train(... mosaic=1.0, mixup=0.2)
   ```

---

## 📁 九、导出模型（ONNX、TensorRT等）

```python
model.export(format="onnx")     # 导出 ONNX 模型
model.export(format="engine")   # 导出 TensorRT 引擎（需 CUDA）
```

---

## 🧩 十、完整训练命令汇总（CLI 版本）

```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16
```

---

## ✅ 总结

| 步骤 | 工具/方法 |
|------|-----------|
| 数据处理 | Python 脚本（XML转TXT） |
| 数据划分 | 自定义脚本或 `sklearn.model_selection.train_test_split` |
| 模型训练 | YOLOv8 Detect 模型 |
| 推理 | `model.predict()` |
| 评估 | `model.val()` |
| 部署 | `model.export()` |

---

以上文字及代码仅供参考。
