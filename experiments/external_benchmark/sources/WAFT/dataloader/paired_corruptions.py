"""Paired clean/corrupt Spring samples with shared geometry."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.augmentor import FlowAugmentor
from dataloader.flow.spring import Spring

CORRUPTIONS = (
    "brightness",
    "contrast",
    "saturate",
    "defocus_blur",
    "gaussian_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_noise",
    "impulse_noise",
    "speckle_noise",
    "shot_noise",
    "pixelate",
    "jpeg_compression",
    "elastic_transform",
    "spatter",
    "frost",
    "snow",
    "rain",
    "fog",
)
GROUPS = {
    "brightness": "color",
    "contrast": "color",
    "saturate": "color",
    "defocus_blur": "blur",
    "gaussian_blur": "blur",
    "glass_blur": "blur",
    "motion_blur": "blur",
    "zoom_blur": "blur",
    "gaussian_noise": "noise",
    "impulse_noise": "noise",
    "speckle_noise": "noise",
    "shot_noise": "noise",
    "pixelate": "quality",
    "jpeg_compression": "quality",
    "elastic_transform": "quality",
    "spatter": "weather",
    "frost": "weather",
    "snow": "weather",
    "rain": "weather",
    "fog": "weather",
}
GROUP_NAMES = ("color", "blur", "noise", "quality", "weather")
TRAIN_CORRUPTIONS = tuple(name for name in CORRUPTIONS if name != "elastic_transform")


def _clip(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def _kernel_size(severity: int) -> int:
    return 2 * max(1, severity) + 1


def _blur(image, name: str, severity: int, rng: np.random.Generator):
    if name in {"defocus_blur", "gaussian_blur", "glass_blur"}:
        result = cv2.GaussianBlur(image, (_kernel_size(severity),) * 2, 0)
        if name == "glass_blur":
            h, w = image.shape[:2]
            displacement = rng.integers(-severity, severity + 1, size=(h, w, 2), dtype=np.int32)
            yy, xx = np.mgrid[:h, :w]
            sx = np.clip(xx + displacement[..., 0], 0, w - 1)
            sy = np.clip(yy + displacement[..., 1], 0, h - 1)
            result = result[sy, sx]
        return result
    if name == "motion_blur":
        size = _kernel_size(severity) * 2 - 1
        kernel = np.zeros((size, size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[size // 2, :] = 1.0 / size
        else:
            np.fill_diagonal(kernel, 1.0 / size)
        return cv2.filter2D(image, -1, kernel)
    scales = np.linspace(1.0, 1.0 + 0.04 * severity, 3 + severity)
    h, w = image.shape[:2]
    accum = np.zeros_like(image, dtype=np.float32)
    for scale in scales:
        resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        y0 = max(0, (resized.shape[0] - h) // 2)
        x0 = max(0, (resized.shape[1] - w) // 2)
        accum += resized[y0:y0 + h, x0:x0 + w]
    return _clip(accum / len(scales))


def _weather_overlay(shape, name: str, severity: int, rng: np.random.Generator):
    h, w = shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w, 1), dtype=np.float32)
    if name == "rain":
        count = max(40, h * w // 7000 * severity)
        for _ in range(count):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            length = int(rng.integers(8, 20 + 8 * severity))
            cv2.line(overlay, (x, y), (x - length // 3, min(h - 1, y + length)), (220, 230, 240), 1)
            cv2.line(alpha, (x, y), (x - length // 3, min(h - 1, y + length)), (0.35,), 1)
    elif name == "snow":
        count = max(100, h * w // 3000 * severity)
        for _ in range(count):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            radius = int(rng.integers(1, max(2, severity + 2)))
            cv2.circle(overlay, (x, y), radius, (245, 245, 245), -1)
            cv2.circle(alpha, (x, y), radius, (0.65,), -1)
    elif name in {"spatter", "frost"}:
        count = max(20, h * w // 16000 * severity)
        for _ in range(count):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            rx = int(rng.integers(3, 15 + 5 * severity))
            ry = int(rng.integers(3, 15 + 5 * severity))
            color = (185, 205, 220) if name == "frost" else (80, 100, 145)
            a = 0.18 if name == "frost" else 0.3
            cv2.ellipse(overlay, (x, y), (rx, ry), 0, 0, 360, color, -1)
            cv2.ellipse(alpha, (x, y), (rx, ry), 0, 0, 360, (a,), -1)
    else:
        # This is intentionally marked as an approximation in metadata.
        low = cv2.resize(rng.random((max(2, h // 32), max(2, w // 32))).astype(np.float32), (w, h))
        alpha = (0.08 + 0.14 * severity * cv2.GaussianBlur(low, (0, 0), 8))[..., None]
        overlay[:] = 255.0
    return overlay, np.clip(alpha, 0, 0.85)


def apply_corruption(image1, image2, name: str, severity: int, seed: int):
    """Apply one temporally shared corruption to a pair."""
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption: {name}")
    if name == "elastic_transform":
        raise ValueError("elastic_transform is excluded until its flow Jacobian is implemented")
    rng = np.random.default_rng(seed)
    severity = int(np.clip(severity, 1, 5))
    a, b = image1.astype(np.float32), image2.astype(np.float32)

    if name == "brightness":
        factor = 1.0 + 0.12 * severity
        a, b = a * factor, b * factor
    elif name == "contrast":
        factor = 1.0 + 0.15 * severity
        a, b = (a - 127.5) * factor + 127.5, (b - 127.5) * factor + 127.5
    elif name == "saturate":
        factor = 1.0 + 0.18 * severity
        for original in (a, b):
            hsv = cv2.cvtColor(_clip(original), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
            converted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if original is a:
                a = converted.astype(np.float32)
            else:
                b = converted.astype(np.float32)
    elif name in {"defocus_blur", "gaussian_blur", "glass_blur", "motion_blur", "zoom_blur"}:
        a, b = _blur(_clip(a), name, severity, rng), _blur(_clip(b), name, severity, rng)
    elif name == "gaussian_noise":
        sigma = 5.0 * severity
        a += rng.normal(0, sigma, a.shape)
        b += rng.normal(0, sigma, b.shape)
    elif name == "impulse_noise":
        probability = 0.01 * severity
        for value in (a, b):
            mask = rng.random(value.shape[:2]) < probability
            value[mask] = rng.choice((0, 255), size=(int(mask.sum()), 1))
    elif name == "speckle_noise":
        strength = 0.04 * severity
        a += a * rng.normal(0, strength, a.shape)
        b += b * rng.normal(0, strength, b.shape)
    elif name == "shot_noise":
        scale = max(1.0, 30.0 / severity)
        a = rng.poisson(np.clip(a, 0, 255) / scale) * scale
        b = rng.poisson(np.clip(b, 0, 255) / scale) * scale
    elif name == "pixelate":
        h, w = a.shape[:2]
        factor = max(2, 1 + severity)
        size = (max(1, w // factor), max(1, h // factor))
        a = cv2.resize(cv2.resize(_clip(a), size, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(cv2.resize(_clip(b), size, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)
    elif name == "jpeg_compression":
        quality = max(10, 85 - 14 * severity)
        encoded_a = cv2.imencode(".jpg", _clip(a), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        encoded_b = cv2.imencode(".jpg", _clip(b), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        a = cv2.cvtColor(cv2.imdecode(encoded_a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(cv2.imdecode(encoded_b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        overlay, alpha = _weather_overlay(a.shape, name, severity, rng)
        a = a * (1 - alpha) + overlay * alpha
        b = b * (1 - alpha) + overlay * alpha
    return _clip(a), _clip(b), {"seed": int(seed), "approximation": name == "fog"}


class PairedSpringCorruptionDataset(Dataset):
    """Load raw Spring pairs, share geometry, then corrupt only the image copy."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_size=(540, 960),
        min_scale=-1.0,
        max_scale=-0.8,
        do_flip: bool = True,
        paired: bool = True,
        corruption_names: tuple[str, ...] | list[str] | None = None,
        scene_names: tuple[str, ...] | list[str] | None = None,
        severity_min: int = 1,
        severity_max: int = 4,
        seed: int = 42,
        balanced_groups: bool = False,
    ):
        self.base = Spring(aug_params=None, root=str(root), split=split)
        if isinstance(scene_names, str):
            scene_names = tuple(value for value in scene_names.split(",") if value)
        allowed_scenes = set(scene_names or ())
        self.indices = [
            index for index, info in enumerate(self.base.extra_info)
            if not allowed_scenes or info[1] in allowed_scenes
        ]
        if not self.indices:
            raise ValueError("scene_names selects no Spring samples")
        self.spatial = FlowAugmentor(
            crop_size=list(crop_size), min_scale=min_scale, max_scale=max_scale, do_flip=do_flip
        )
        self.paired = paired
        self.corruption_names = tuple(corruption_names or TRAIN_CORRUPTIONS)
        unknown = set(self.corruption_names) - set(TRAIN_CORRUPTIONS)
        if unknown:
            raise ValueError(f"corruptions unavailable for supervised training: {sorted(unknown)}")
        self.severity_min = int(severity_min)
        self.severity_max = int(severity_max)
        self.seed = int(seed)
        self.balanced_groups = bool(balanced_groups)
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        base_index = self.indices[int(index)]
        seed = self.seed + self.epoch * max(1, len(self)) + base_index
        np_state, py_state = np.random.get_state(), random.getstate()
        np.random.seed(seed & 0xFFFFFFFF)
        random.seed(seed)
        try:
            image1, image2, flow, valid = self.base.fetch(base_index)
            image1 = image1.permute(1, 2, 0).numpy().astype(np.uint8)
            image2 = image2.permute(1, 2, 0).numpy().astype(np.uint8)
            flow = flow.permute(1, 2, 0).numpy().astype(np.float32)
            valid = valid.numpy().astype(bool)
            image1, image2, flow, valid = self.spatial.spatial_transform(image1, image2, flow, valid)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

        clean1, clean2 = np.ascontiguousarray(image1), np.ascontiguousarray(image2)
        if self.paired:
            rng = np.random.default_rng(seed ^ 0x5DEECE66D)
            if self.balanced_groups:
                groups = tuple(group for group in GROUP_NAMES if any(GROUPS[name] == group for name in self.corruption_names))
                group = groups[int(rng.integers(len(groups)))]
                names = tuple(name for name in self.corruption_names if GROUPS[name] == group)
                name = names[int(rng.integers(len(names)))]
            else:
                name = self.corruption_names[int(rng.integers(len(self.corruption_names)))]
            severity = int(rng.integers(self.severity_min, self.severity_max + 1))
            corrupt1, corrupt2, extra = apply_corruption(clean1, clean2, name, severity, seed)
        else:
            name, severity, corrupt1, corrupt2, extra = "clean", 0, clean1, clean2, {"seed": seed}
        group = GROUPS.get(name, "clean")
        return {
            "clean_image1": torch.from_numpy(clean1).permute(2, 0, 1).float(),
            "clean_image2": torch.from_numpy(clean2).permute(2, 0, 1).float(),
            "corrupt_image1": torch.from_numpy(np.ascontiguousarray(corrupt1)).permute(2, 0, 1).float(),
            "corrupt_image2": torch.from_numpy(np.ascontiguousarray(corrupt2)).permute(2, 0, 1).float(),
            "flow": torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1).float(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).float(),
            "corruption_id": CORRUPTIONS.index(name) if name in CORRUPTIONS else -1,
            "corruption_name": name,
            "corruption_group": group,
            "group_id": GROUP_NAMES.index(group) if group in GROUP_NAMES else -1,
            "severity": severity,
            "sample_id": f"{self.base.extra_info[base_index][1]}_{self.base.extra_info[base_index][2]}_{self.base.extra_info[base_index][0]:04d}_{self.base.extra_info[base_index][3]}",
            "scene_id": self.base.extra_info[base_index][1],
            "transform_meta": {"seed": int(extra["seed"]), "approximation": bool(extra.get("approximation", False))},
        }
