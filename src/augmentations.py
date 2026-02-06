"""
Data augmentation transforms for flood segmentation.
Applies consistent transforms to all inputs (post, pre, infra, label).
"""
import numpy as np
import random


class FloodAugmentation:
    """
    Augmentation pipeline for flood segmentation.
    Applies geometric and intensity transforms consistently across all inputs.
    """
    def __init__(self, 
                 flip_prob=0.5, 
                 rotate_prob=0.5,
                 brightness_prob=0.3,
                 noise_prob=0.2):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.brightness_prob = brightness_prob
        self.noise_prob = noise_prob
    
    def __call__(self, sample):
        """
        Apply augmentations to sample dict.
        Sample contains: image, pre_image, infra, label (all numpy arrays C,H,W)
        """
        image = sample["image"]
        label = sample["label"]
        pre_image = sample.get("pre_image")
        infra = sample.get("infra")
        
        # --- Geometric Augmentations (apply to ALL) ---
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
            if pre_image is not None:
                pre_image = np.flip(pre_image, axis=2).copy()
            if infra is not None:
                infra = np.flip(infra, axis=2).copy()
        
        # Vertical flip
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
            if pre_image is not None:
                pre_image = np.flip(pre_image, axis=1).copy()
            if infra is not None:
                infra = np.flip(infra, axis=1).copy()
        
        # 90-degree rotations (0, 90, 180, 270)
        if random.random() < self.rotate_prob:
            k = random.randint(1, 3)  # 1=90°, 2=180°, 3=270°
            image = np.rot90(image, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(1, 2)).copy()
            if pre_image is not None:
                pre_image = np.rot90(pre_image, k, axes=(1, 2)).copy()
            if infra is not None:
                infra = np.rot90(infra, k, axes=(1, 2)).copy()
        
        # --- Intensity Augmentations (SAR images only, NOT labels) ---
        
        # Brightness/contrast adjustment
        if random.random() < self.brightness_prob:
            # Small brightness shift
            brightness = random.uniform(-0.1, 0.1)
            contrast = random.uniform(0.9, 1.1)
            
            image = np.clip(image * contrast + brightness, 0, 1)
            if pre_image is not None and pre_image.shape[0] > 0:
                pre_image = np.clip(pre_image * contrast + brightness, 0, 1)
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.randn(*image.shape).astype(np.float32) * noise_level
            image = np.clip(image + noise, 0, 1)
            if pre_image is not None and pre_image.shape[0] > 0:
                pre_noise = np.random.randn(*pre_image.shape).astype(np.float32) * noise_level
                pre_image = np.clip(pre_image + pre_noise, 0, 1)
        
        # Update sample
        sample["image"] = image.astype(np.float32)
        sample["label"] = label.astype(np.float32)
        if pre_image is not None:
            sample["pre_image"] = pre_image.astype(np.float32)
        if infra is not None:
            sample["infra"] = infra.astype(np.float32)
        
        return sample


class RandomCrop:
    """Random crop augmentation for training diversity."""
    def __init__(self, crop_size=448, prob=0.3):
        self.crop_size = crop_size
        self.prob = prob
    
    def __call__(self, sample):
        if random.random() > self.prob:
            return sample
        
        image = sample["image"]
        _, h, w = image.shape
        
        if h <= self.crop_size or w <= self.crop_size:
            return sample
        
        # Random crop position
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        
        # Crop all arrays
        for key in ["image", "label", "pre_image", "infra"]:
            if key in sample and sample[key] is not None and sample[key].shape[0] > 0:
                sample[key] = sample[key][:, top:top+self.crop_size, left:left+self.crop_size].copy()
        
        return sample


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transforms():
    """Get training augmentation pipeline."""
    return Compose([
        RandomCrop(crop_size=256, prob=1.0),  # Always crop to 256x256 to save memory
        FloodAugmentation(
            flip_prob=0.5,
            rotate_prob=0.5,
            brightness_prob=0.3,
            noise_prob=0.2
        )
    ])


def get_val_transforms():
    """Validation: no augmentation."""
    return None
