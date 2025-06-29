# Pavement-Texture-Monocular-Depth-Estimation

This project is qualified for participation in the **"National Pioneer Cup on Intelligent Computing – Shandong University of Science and Technology Selection."**

Monocular depth estimation of asphalt pavement texture using GAN and pretrained Depth-Anything-V2. Includes data preprocessing, augmentation, dual-branch generator, multi-loss optimization, and evaluation.

---

## 📌 Highlights

- 🔍 Integrates Depth-Anything-V2 pretrained model for auxiliary supervision  
- 🧠 GAN architecture with dual-branch generator and PatchGAN discriminator  
- 🧪 Composite loss: BerHu, L1, SSIM, Gradient, Perceptual (VGG19), Scale-Invariant  
- 🧰 Data preprocessing and enhancement: flipping, rotation, scaling, translation  
- 📊 Evaluation metrics: MAE, RMSE, relative error  

---

## 🖼️ Model Architecture

![Model Structure](image/Model_Structure.png)

> The model features a GAN with a dual-branch generator and PatchGAN discriminator, guided by multi-loss functions for accurate depth estimation.

---

## 🎯 Results

### Quantitative Evaluation

![Result](image/results.png)  

| Metric         | Value          |
| -------------- | -------------- |
| MAE            | 0.4008 mm     |
| RMSE           | 0.4919 mm     |
| Relative Error | 116.5508%     |


> These results demonstrate the effectiveness of the GAN-based depth estimation enhanced by Depth-Anything-V2.

---

## 🗂️ Project Structure

```
├── Code.py                   # Main training script including data loading, model definition, and training process
├── requirements.txt          # List of required Python packages
├── image/                    # Folder containing images
│   ├── Model_Structure.png   # Model architecture diagram
│   └── results.png           # Model prediction results visualization
└── README.md                 # Project documentation

```

---

## ⚙️ Setup & Usage

### 1. Clone Repository
bash
git clone [https://github.com/rxxxx26/pavement-depth-estimation.git](https://github.com/rxxxx26/Pavement-Texture-Monocular-Depth-Estimation.git)
cd pavement-depth-estimation


### 2. Install Dependencies
bash
pip install -r requirements.txt


### 3. Prepare Data
- Place RGB .jpg files in data/RGB/
- Place depth100.mat file in data/
- Download Depth-Anything-V2 weights from [official repo](https://github.com/DepthAnything/Depth-Anything-V2) and place them locally (update path in Code.py accordingly)

### 4. Start Training
bash
python Code.py
