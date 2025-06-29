import tensorflow as tf
import numpy as np
import pathlib
import os
import pandas as pd
import re
import h5py
import scipy.io
import torch
import sys
from tensorflow.keras.layers import Layer, Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, LeakyReLU, Dense, Flatten, Add, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# Add the Depth-Anything-V2 directory to the Python path
sys.path.append(r'C:\Users\23093\Depth-Anything-V2')  # Corrected path
from depth_anything_v2.dpt import DepthAnythingV2

# Custom ResizeLayer
class ResizeLayer(Layer):
    def __init__(self, target_size, method='bilinear', **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size
        self.method = method

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method=self.method)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({"target_size": self.target_size, "method": self.method})
        return config

# Custom Training Progress Callback
class TrainingProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        print(f"\nEpoch {epoch + 1}/20 Starting...")

    def on_batch_end(self, batch, logs=None):
        print(f"Epoch {self.epoch + 1}/20 - Batch {batch + 1}/{steps_per_epoch} - Loss: {logs.get('loss'):.4f}, MSE: {logs.get('mse'):.4f}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/20 Ended - Loss: {logs.get('loss'):.4f}, Val MSE: {logs.get('val_mse'):.4f}")

# Check TensorFlow setup
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('float32')

# Paths
rgb_path = r"C:\Users\23093\Desktop\RGB-D\RGB"
depth_mat_path = r"C:\Users\23093\Desktop\RGB-D\depth100.mat"

# 检查路径是否存在
if not os.path.exists(rgb_path):
    print(f"Error: RGB directory '{rgb_path}' does not exist.")
    exit(1)
else:
    print(f"RGB directory found at '{rgb_path}'.")

if not os.path.exists(depth_mat_path):
    print(f"Error: Depth file '{depth_mat_path}' does not exist.")
    exit(1)
else:
    print(f"Depth file found at '{depth_mat_path}'.")

# Load and sort RGB image paths
picture_root = pathlib.Path(rgb_path)
all_pic_path = [str(path) for path in picture_root.glob('*.[jJ][pP][gG]')]  # 支持 .jpg 和 .jpeg
def extract_number(filename):
    match = re.search(r'(\d+)\.[jJ][pP][gG]$', filename)
    return int(match.group(1)) if match else 0
all_pic_path.sort(key=extract_number)
print(f"Number of RGB images: {len(all_pic_path)}")
if len(all_pic_path) == 0:
    print("No .jpg or .jpeg files found in the RGB directory. Please check the file extensions.")
    exit(1)

# Load depth maps from depth100.mat
try:
    with h5py.File(depth_mat_path, 'r') as f:
        print("Successfully loaded depth100.mat with h5py!")
        if 'depths' in f.keys():
            all_depths = np.array(f['depths'])
        else:
            for key in f.keys():
                if not key.startswith('__'):
                    all_depths = np.array(f[key])
                    break
            else:
                raise KeyError("No suitable depth data found in depth100.mat")

except Exception as e:
    print(f"Failed to load depth100.mat with h5py: {e}")
    print("Attempting to load with scipy.io.loadmat (for MATLAB v7 or earlier format)...")
    
    try:
        mat_data = scipy.io.loadmat(depth_mat_path)
        print("Successfully loaded depth100.mat with scipy.io.loadmat!")
        print("Keys in depth100.mat:", list(mat_data.keys()))
        
        depth_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                depth_key = key
                break
        
        if depth_key is None:
            raise KeyError("No suitable depth data found in depth100.mat")
        
        all_depths = mat_data[depth_key]
        print(f"Loaded depth data from key '{depth_key}'")

    except Exception as e:
        print(f"Error loading depth100.mat with scipy.io.loadmat: {e}")
        print("The file may be corrupted or in an unsupported format.")
        print("Please try opening the file in MATLAB and re-saving it with '-v7.3' flag (HDF5 format).")
        print("In MATLAB, use: save('depth100_new.mat', 'depths', '-v7.3')")
        exit(1)

# 检查深度图形状
print(f"Depth map shape (raw): {all_depths.shape}")

# 如果深度图形状是 (H, W, N) 或 (1, H, W, N)，转置为 (N, H, W) 或 (N, H, W, 1)
if all_depths.ndim == 4 and all_depths.shape[0] == 1:
    all_depths = all_depths[0]  # 去掉第一个维度
if all_depths.ndim == 3 and all_depths.shape[-1] in [100, all_depths.shape[0], all_depths.shape[1]]:  # 检查最后一维是否为样本数量
    all_depths = np.transpose(all_depths, (2, 0, 1))  # 转置为 (N, H, W)，即 (100, 208, 146)
elif all_depths.ndim == 4 and all_depths.shape[-1] == 1:
    all_depths = all_depths[..., 0]  # 去掉通道维度，变为 (N, H, W)

print(f"Depth map shape (adjusted): {all_depths.shape}")

# 清洗深度图数据
all_depths = all_depths.astype(np.float32)
all_depths = np.nan_to_num(all_depths, nan=0.0, posinf=0.0, neginf=0.0)

# 使用分位数裁剪深度值
depths_flat = all_depths.flatten()
lower_bound = np.percentile(depths_flat, 1)  # 1% 分位数
upper_bound = np.percentile(depths_flat, 99)  # 99% 分位数
print("Depth data statistics (before clipping) - Min:", np.min(all_depths), "Max:", np.max(all_depths), "Mean:", np.mean(all_depths))
print(f"1% percentile: {lower_bound:.2f} mm, 99% percentile: {upper_bound:.2f} mm")

# 假设沥青路面纹理深度范围为 0-5 mm，结合分位数裁剪
all_depths = np.clip(all_depths, max(0, lower_bound), min(5.0, upper_bound))
print("Depth data statistics (after clipping) - Min:", np.min(all_depths), "Max:", np.max(all_depths), "Mean:", np.mean(all_depths))

# Align sample count
min_count = min(len(all_pic_path), all_depths.shape[0])
all_pic_path = all_pic_path[:min_count]
all_depths = all_depths[:min_count]
print(f"Adjusted to match: {min_count} samples")

# 对齐深度图分辨率与 RGB 图像（256x256）
all_depths = tf.convert_to_tensor(all_depths, dtype=tf.float32)
all_depths = tf.expand_dims(all_depths, axis=3)  # 形状: (N, H, W, 1)
all_depths = tf.image.resize(all_depths, [256, 256], method='bilinear')  #  shapes: (N, 256, 256, 1)
# 归一化深度图
all_depths = (all_depths - tf.reduce_min(all_depths)) / (tf.reduce_max(all_depths) - tf.reduce_min(all_depths) + 1e-7)

# 加载 Depth-Anything-V2-Small 模型
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Load Depth-Anything-V2-Small
encoder = 'vits'  # Use the Small model
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(r'C:\Users\23093\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth', map_location='cpu'))
depth_anything.eval()

# Precompute depth maps using Depth-Anything-V2
precomputed_depths = []
for pic_path in tqdm(all_pic_path, desc="Precomputing depth maps with Depth-Anything-V2"):
    # Load and preprocess the RGB image
    image_raw = tf.io.read_file(pic_path)
    image_tensor = tf.image.decode_jpeg(image_raw, channels=3)
    image_tensor = tf.image.resize(image_tensor, [518, 518], method='bilinear')  # Resize to 518x518
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
    image_tensor = tf.transpose(image_tensor, [2, 0, 1])  # (H, W, C) -> (C, H, W)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # (C, H, W) -> (1, C, H, W)
    
    # Convert to PyTorch tensor
    image_np = image_tensor.numpy()
    image_torch = torch.from_numpy(image_np).float()
    
    # Run Depth-Anything-V2
    with torch.no_grad():
        depth_map = depth_anything(image_torch)  # Expected shape: [1, 1, 518, 518] or [1, 518, 518]
    
    # Ensure the depth map has the correct shape
    depth_map = depth_map.cpu().numpy()  # Convert to NumPy
    if depth_map.ndim == 4:  # Shape: [1, 1, 518, 518]
        depth_map = depth_map.squeeze(0)  # Shape: [1, 518, 518]
    elif depth_map.ndim == 3:  # Shape: [1, 518, 518]
        pass  # Already in the correct shape
    else:  # Shape: [518, 518] or other
        depth_map = depth_map[np.newaxis, :, :]  # Shape: [1, 518, 518]
    
    # Normalize to [0, 1]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-7)
    precomputed_depths.append(depth_map)

# Convert precomputed depth maps to TensorFlow tensor
precomputed_depths = np.stack(precomputed_depths)  # Shape: [N, 1, 518, 518]
precomputed_depths = tf.convert_to_tensor(precomputed_depths, dtype=tf.float32)
# Ensure the channel dimension is correct
if precomputed_depths.shape[1] == 1:
    precomputed_depths = tf.transpose(precomputed_depths, [0, 2, 3, 1])  # Shape: [N, 518, 518, 1]
precomputed_depths = tf.image.resize(precomputed_depths, [64, 64], method='bilinear')  # Shape: [N, 64, 64, 1]

# 数据增强：通过反转、旋转、平移和缩放增加数据量
augmented_pic_paths = []
augmented_depths = []
augmented_precomputed_depths = []
for i in range(min_count):
    pic_path = all_pic_path[i]
    depth = all_depths[i]
    pre_depth = precomputed_depths[i]  # Shape: [64, 64, 1]
    pre_depth = tf.expand_dims(pre_depth, axis=0)  # Shape: [1, 64, 64, 1]
    print("pre_depth shape:", pre_depth.shape)  # Debug print

    # 原始数据
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(depth)
    augmented_precomputed_depths.append(tf.squeeze(pre_depth, axis=0))  # Shape: [64, 64, 1]

    # 水平翻转
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(tf.image.flip_left_right(depth))
    augmented_precomputed_depths.append(tf.squeeze(tf.image.flip_left_right(pre_depth), axis=0))

    # 垂直翻转
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(tf.image.flip_up_down(depth))
    augmented_precomputed_depths.append(tf.squeeze(tf.image.flip_up_down(pre_depth), axis=0))

    # 旋转 90 度
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(tf.image.rot90(depth, k=1))
    augmented_precomputed_depths.append(tf.squeeze(tf.image.rot90(pre_depth, k=1), axis=0))

    # 旋转 180 度
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(tf.image.rot90(depth, k=2))
    augmented_precomputed_depths.append(tf.squeeze(tf.image.rot90(pre_depth, k=2), axis=0))

    # 旋转 270 度
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(tf.image.rot90(depth, k=3))
    augmented_precomputed_depths.append(tf.squeeze(tf.image.rot90(pre_depth, k=3), axis=0))

    # 平移（随机平移 10% 的图像尺寸）
    shift_x = tf.random.uniform([], -0.1, 0.1) * 256
    shift_y = tf.random.uniform([], -0.1, 0.1) * 256
    translated_depth = tf.keras.layers.Lambda(lambda x: tf.roll(x, shift=[int(shift_y), int(shift_x)], axis=[1, 2]))(depth)
    translated_pre_depth = tf.keras.layers.Lambda(lambda x: tf.roll(x, shift=[int(shift_y), int(shift_x)], axis=[1, 2]))(pre_depth)
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(translated_depth)
    augmented_precomputed_depths.append(tf.squeeze(translated_pre_depth, axis=0))

    # 缩放（随机缩放 0.9-1.1 倍）
    scale = tf.random.uniform([], 0.9, 1.1)
    scaled_depth = tf.image.resize(depth, [int(256 * scale), int(256 * scale)], method='bilinear')
    scaled_depth = tf.image.resize_with_crop_or_pad(scaled_depth, 256, 256)
    print("scale:", scale, "scaled_pre_depth shape before resize:", pre_depth.shape)  # Debug print
    scaled_pre_depth = tf.image.resize(pre_depth, [int(64 * scale), int(64 * scale)], method='bilinear')
    print("scaled_pre_depth shape after resize:", scaled_pre_depth.shape)  # Debug print
    scaled_pre_depth = tf.image.resize_with_crop_or_pad(scaled_pre_depth, 64, 64)
    augmented_pic_paths.append(pic_path)
    augmented_depths.append(scaled_depth)
    augmented_precomputed_depths.append(tf.squeeze(scaled_pre_depth, axis=0))

# 转换为 TensorFlow 张量
augmented_pic_paths = tf.convert_to_tensor(augmented_pic_paths, dtype=tf.string)
augmented_depths = tf.stack(augmented_depths)
augmented_precomputed_depths = tf.stack(augmented_precomputed_depths)
print(f"Number of augmented samples: {len(augmented_pic_paths)}")

# Enhanced data augmentation
def augment_both_train(image_path, depth_pic, pre_depth):
    try:
        image_raw = tf.io.read_file(image_path)
        image_tensor = tf.image.decode_jpeg(image_raw, channels=3)
    except tf.errors.InvalidArgumentError as e:
        print(f"Warning: Failed to decode JPEG file {image_path}: {e}. Skipping this file.")
        return None, None, None

    image_tensor = tf.image.resize(image_tensor, [256, 256], antialias=True)
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0

    depth_pic = tf.image.resize(depth_pic, [64, 64], method='bilinear', antialias=True)
    depth_pic = tf.clip_by_value(depth_pic, 0.0, 1.0)

    pre_depth = tf.image.resize(pre_depth, [64, 64], method='bilinear', antialias=True)
    pre_depth = tf.clip_by_value(pre_depth, 0.0, 1.0)

    image_tensor = tf.where(tf.math.is_nan(image_tensor) | tf.math.is_inf(image_tensor), tf.zeros_like(image_tensor), image_tensor)
    depth_pic = tf.where(tf.math.is_nan(depth_pic) | tf.math.is_inf(depth_pic), tf.zeros_like(depth_pic), depth_pic)
    pre_depth = tf.where(tf.math.is_nan(pre_depth) | tf.math.is_inf(pre_depth), tf.zeros_like(pre_depth), pre_depth)
    image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)
    depth_pic = tf.clip_by_value(depth_pic, 0.0, 1.0)
    pre_depth = tf.clip_by_value(pre_depth, 0.0, 1.0)
    return image_tensor, depth_pic, pre_depth

def preprocess_val(image_path, depth_pic, pre_depth):
    try:
        image_raw = tf.io.read_file(image_path)
        image_tensor = tf.image.decode_jpeg(image_raw, channels=3)
    except tf.errors.InvalidArgumentError as e:
        print(f"Warning: Failed to decode JPEG file {image_path}: {e}. Skipping this file.")
        return None, None, None

    image_tensor = tf.image.resize(image_tensor, [256, 256], antialias=True)
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
    depth_pic = tf.image.resize(depth_pic, [64, 64], method='bilinear', antialias=True)
    depth_pic = tf.clip_by_value(depth_pic, 0.0, 1.0)
    pre_depth = tf.image.resize(pre_depth, [64, 64], method='bilinear', antialias=True)
    pre_depth = tf.clip_by_value(pre_depth, 0.0, 1.0)
    image_tensor = tf.where(tf.math.is_nan(image_tensor) | tf.math.is_inf(image_tensor), tf.zeros_like(image_tensor), image_tensor)
    depth_pic = tf.where(tf.math.is_nan(depth_pic) | tf.math.is_inf(depth_pic), tf.zeros_like(depth_pic), depth_pic)
    pre_depth = tf.where(tf.math.is_nan(pre_depth) | tf.math.is_inf(pre_depth), tf.zeros_like(pre_depth), pre_depth)
    return image_tensor, depth_pic, pre_depth

# Dataset creation
dataset = tf.data.Dataset.from_tensor_slices((augmented_pic_paths, augmented_depths, augmented_precomputed_depths))
train_ratio = 0.8
train_count = int(len(augmented_pic_paths) * train_ratio)
test_count = len(augmented_pic_paths) - train_count
train_dataset = dataset.skip(test_count).map(augment_both_train, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset.take(test_count).map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

# 过滤掉 None 值
train_dataset = train_dataset.filter(lambda x, y, z: x is not None and y is not None and z is not None)
test_dataset = test_dataset.filter(lambda x, y, z: x is not None and y is not None and z is not None)

BUFFER_SIZE = 1000
BATCH_SIZE = 4  # 调整为 4 以减少 CPU 内存压力
dataset_train = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
dataset_test = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
steps_per_epoch = max(train_count // BATCH_SIZE, 1)
validation_steps = max(test_count // BATCH_SIZE, 1)

# 自定义生成器模型，适配预计算的深度图
def generator_model():
    # Inputs: RGB image and precomputed depth map
    rgb_input = Input(shape=(256, 256, 3), name='rgb_input')
    depth_input = Input(shape=(64, 64, 1), name='depth_input')

    # Process RGB input
    x_rgb = Conv2D(64, (3, 3), padding='same')(rgb_input)
    x_rgb = BatchNormalization()(x_rgb)
    x_rgb = tf.keras.activations.gelu(x_rgb)
    x_rgb = Conv2D(64, (3, 3), padding='same')(x_rgb)
    x_rgb = BatchNormalization()(x_rgb)
    x_rgb = tf.keras.activations.gelu(x_rgb)
    x_rgb = tf.image.resize(x_rgb, [64, 64], method='bilinear')

    # Process depth input
    x_depth = Conv2D(64, (3, 3), padding='same')(depth_input)
    x_depth = BatchNormalization()(x_depth)
    x_depth = tf.keras.activations.gelu(x_depth)

    # Concatenate RGB and depth features
    x = Concatenate()([x_rgb, x_depth])  # Shape: [batch, 64, 64, 128]

    # Further processing
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    x = Dropout(0.1)(x)

    # 上采样到目标分辨率
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    x = Dropout(0.1)(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.gelu(x)
    x = Dropout(0.1)(x)

    # 调整到目标输出分辨率 64x64
    x = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    x = ResizeLayer(target_size=[64, 64])(x)

    return Model(inputs=[rgb_input, depth_input], outputs=x)

# Enhanced Discriminator Model
def discriminator_model():
    inputs = Input(shape=(64, 64, 1))
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv2D(512, (4, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (4, 4), padding='same', activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)

generator = generator_model()
discriminator = discriminator_model()

# Multi-layer Perceptual Loss
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg.trainable = False
loss_layers = ['block3_conv3', 'block4_conv4', 'block5_conv4']
outputs = [vgg.get_layer(layer).output for layer in loss_layers]
loss_model = Model(inputs=vgg.input, outputs=outputs)

def perceptual_loss(true_img, pred_depth):
    pred_depth_rgb = tf.image.grayscale_to_rgb(pred_depth)
    true_img = tf.image.resize(true_img, [256, 256])
    pred_depth_rgb = tf.image.resize(pred_depth_rgb, [256, 256])
    true_feats = loss_model(true_img)
    pred_feats = loss_model(pred_depth_rgb)
    return tf.reduce_mean([tf.reduce_mean(tf.square(t - p)) for t, p in zip(true_feats, pred_feats)])

# Enhanced Loss Functions
def berhu(true, pred):
    diff = tf.abs(true - pred)
    c = 0.2 * tf.reduce_max(diff)
    c = tf.maximum(c, 1e-7)
    return tf.reduce_mean(tf.where(diff <= c, diff, (diff**2 + c**2)/(2*c)))

def gradient_loss(true, pred):
    true_grad = tf.image.sobel_edges(true)
    pred_grad = tf.image.sobel_edges(pred)
    return tf.reduce_mean(tf.abs(true_grad - pred_grad))

def scale_invariant_loss(true, pred):
    log_true = tf.math.log(true + 1e-7)
    log_pred = tf.math.log(pred + 1e-7)
    diff = log_true - log_pred
    return tf.reduce_mean(diff**2) - (tf.reduce_mean(diff)**2)

# 调整损失函数权重，增加 SSIM 和梯度损失的权重以关注纹理细节
def combined_loss(true_depth, pred_depth, true_img):
    berhu_loss = berhu(true_depth, pred_depth)
    l1_loss = tf.reduce_mean(tf.abs(true_depth - pred_depth))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(true_depth, pred_depth, max_val=1.0))
    grad_loss = gradient_loss(true_depth, pred_depth)
    perc_loss = perceptual_loss(true_img, pred_depth)
    si_loss = scale_invariant_loss(true_depth, pred_depth)
    return 0.15*berhu_loss + 0.15*l1_loss + 0.35*ssim_loss + 0.2*grad_loss + 0.05*perc_loss + 0.1*si_loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, true_depth, pred_depth, true_img):
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    depth_loss = combined_loss(true_depth, pred_depth, true_img)
    return 0.01 * gan_loss + 0.99 * depth_loss

# Optimizers with gradient clipping and learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=1000,
    decay_rate=0.9
)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999)

# Training Step with Gradient Clipping
@tf.function
def train_step(images, true_depths, pre_depths):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_depths = generator([images, pre_depths], training=True)
        real_output = discriminator(true_depths, training=True)
        fake_output = discriminator(generated_depths, training=True)
        gen_loss = generator_loss(fake_output, true_depths, generated_depths, images)
        disc_loss = discriminator_loss(real_output, fake_output)
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gen_grads]
    disc_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in disc_grads]
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Custom Pretraining Loop for Generator
def pretrain_generator(dataset_train, dataset_test, epochs, steps_per_epoch, validation_steps):
    best_val_mse = float('inf')
    patience = 10
    wait = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} Starting...")
        gen_loss_avg = tf.keras.metrics.Mean()
        mse_metric = tf.keras.metrics.MeanSquaredError()
        
        for step, (images, true_depths, pre_depths) in enumerate(dataset_train.take(steps_per_epoch)):
            with tf.GradientTape() as gen_tape:
                pred_depths = generator([images, pre_depths], training=True)
                loss = combined_loss(true_depths, pred_depths, images)
            gen_grads = gen_tape.gradient(loss, generator.trainable_variables)
            gen_grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gen_grads]
            generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
            gen_loss_avg.update_state(loss)
            mse_metric.update_state(true_depths, pred_depths)
            print(f"Epoch {epoch + 1}/{epochs} - Batch {step + 1}/{steps_per_epoch} - Loss: {loss:.4f}, MSE: {mse_metric.result():.4f}")
        
        val_mse = tf.keras.metrics.MeanSquaredError()
        for val_images, val_depths, val_pre_depths in dataset_test.take(validation_steps):
            pred_depths = generator([val_images, val_pre_depths], training=False)
            val_mse.update_state(val_depths, pred_depths)
        print(f"Epoch {epoch + 1}/{epochs} Ended - Loss: {gen_loss_avg.result():.4f}, Val MSE: {val_mse.result():.4f}")
        
        if val_mse.result() < best_val_mse:
            best_val_mse = val_mse.result()
            wait = 0
            generator.save_weights(r'C:\Users\23093\Desktop\RGB-D\pretrain_best_generator_weights.h5')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered in pretraining. Best Val MSE: {best_val_mse:.4f}")
                break
    
    generator.load_weights(r'C:\Users\23093\Desktop\RGB-D\pretrain_best_generator_weights.h5')

# GAN Training with detailed evaluation
def train(dataset, epochs, target_mse=0.0005):
    best_mse = float('inf')
    patience = 50
    wait = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()
        for images, depths, pre_depths in tqdm(dataset_train.take(steps_per_epoch), desc=f"Epoch {epoch+1}"):
            gen_loss, disc_loss = train_step(images, depths, pre_depths)
            gen_loss_avg.update_state(gen_loss)
            disc_loss_avg.update_state(disc_loss)
        
        # 验证集评估
        val_loss = tf.keras.metrics.Mean()
        val_mse = tf.keras.metrics.Mean()
        val_mae = tf.keras.metrics.Mean()
        val_rmse = tf.keras.metrics.Mean()
        val_rel_error = tf.keras.metrics.Mean()
        for val_images, val_depths, val_pre_depths in dataset_test.take(validation_steps):
            pred = generator([val_images, val_pre_depths], training=False)
            val_loss.update_state(combined_loss(val_depths, pred, val_images))
            val_mse.update_state(tf.reduce_mean(tf.square(val_depths - pred)))
            val_mae.update_state(tf.reduce_mean(tf.abs(val_depths - pred)))
            val_rmse.update_state(tf.sqrt(tf.reduce_mean(tf.square(val_depths - pred))))
            val_rel_error.update_state(tf.reduce_mean(tf.abs(val_depths - pred) / (val_depths + 1e-7)))
        
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Gen Loss: {gen_loss_avg.result():.4f}, "
              f"Disc Loss: {disc_loss_avg.result():.4f}, "
              f"Val Loss: {val_loss.result():.4f}, "
              f"Val MSE: {val_mse.result():.4f}, "
              f"Val MAE: {val_mae.result():.4f}, "
              f"Val RMSE: {val_rmse.result():.4f}, "
              f"Val Relative Error: {val_rel_error.result():.4f}")
        
        if val_mse.result() < best_mse:
            best_mse = val_mse.result()
            wait = 0
            generator.save_weights(r'C:\Users\23093\Desktop\RGB-D\best_generator_weights.h5')
            discriminator.save_weights(r'C:\Users\23093\Desktop\RGB-D\best_discriminator_weights.h5')
        else:
            wait += 1
            if wait >= patience or val_mse.result() <= target_mse:
                print(f"Early stopping triggered. Best Val MSE: {best_mse:.4f}")
                break

    generator.load_weights(r'C:\Users\23093\Desktop\RGB-D\best_generator_weights.h5')
    discriminator.load_weights(r'C:\Users\23093\Desktop\RGB-D\best_discriminator_weights.h5')
    return best_mse

# Run training
pretrain_generator(dataset_train, dataset_test, epochs=20, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
best_mse = train(dataset_train, epochs=100, target_mse=0.0005)
print(f"Final Best MSE: {best_mse:.4f}")

# 最终评估
val_mae = tf.keras.metrics.Mean()
val_rmse = tf.keras.metrics.Mean()
val_rel_error = tf.keras.metrics.Mean()
for val_images, val_depths, val_pre_depths in dataset_test:
    pred = generator([val_images, val_pre_depths], training=False)
    # 反归一化以计算实际误差（以毫米为单位）
    pred = pred * (tf.reduce_max(all_depths) - tf.reduce_min(all_depths)) + tf.reduce_min(all_depths)
    val_depths = val_depths * (tf.reduce_max(all_depths) - tf.reduce_min(all_depths)) + tf.reduce_min(all_depths)
    val_mae.update_state(tf.reduce_mean(tf.abs(val_depths - pred)))
    val_rmse.update_state(tf.sqrt(tf.reduce_mean(tf.square(val_depths - pred))))
    val_rel_error.update_state(tf.reduce_mean(tf.abs(val_depths - pred) / (val_depths + 1e-7)))

print(f"Final Evaluation (in mm) - MAE: {val_mae.result():.2f}, RMSE: {val_rmse.result():.2f}, Relative Error: {val_rel_error.result():.4f}")