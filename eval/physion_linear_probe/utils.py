"""
HamJEPA Physion++ Linear Probe -- Utility Functions

Video-to-frame transforms for HamJEPA evaluation.
HamJEPA's ResNet encoder processes single images, so we decompose
each video clip into individual frames before encoding.
"""

import numpy as np
import torch

# ImageNet normalization (same as HamJEPA training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_physion_frame_transform(
    training: bool = True,
    crop_size: int = 224,
    normalize=(IMAGENET_MEAN, IMAGENET_STD),
    random_horizontal_flip: bool = False,
    **_kwargs,  # ignore unsupported aug params
):
    """
    Create a frame-level transform for HamJEPA Physion++ evaluation.

    Unlike V-JEPA's video-level transform, this operates on individual frames
    extracted from the video clip. Returns the transform callable directly.

    Returns:
        PhysionFrameTransform: callable that takes a video tensor
            [C, T, H, W] and returns a list of frame tensors [[C, H, W], ...]
    """
    return PhysionFrameTransform(
        crop_size=crop_size,
        normalize=normalize,
        training=training,
        random_horizontal_flip=random_horizontal_flip,
    )


class PhysionFrameTransform:
    """
    Converts a video clip tensor into a list of per-frame tensors
    suitable for HamJEPA's image-based ResNet encoder.

    Pipeline per frame:
        Resize(short_side) → CenterCrop(crop_size) → Normalize

    Input:  torch.Tensor [C, T, H, W]  (already decoded, uint8 0-255 or float)
    Output: torch.Tensor [T, C, H, W]  (normalized, ready for batch encoding)
    """

    def __init__(
        self,
        crop_size: int = 224,
        normalize=(IMAGENET_MEAN, IMAGENET_STD),
        training: bool = False,
        random_horizontal_flip: bool = False,
    ):
        self.crop_size = crop_size
        self.short_side = int(crop_size * 256 / 224)
        self.mean = torch.tensor(normalize[0], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(normalize[1], dtype=torch.float32).view(1, 3, 1, 1)
        self.training = training
        self.random_horizontal_flip = random_horizontal_flip

    def __call__(self, buffer):
        """
        Args:
            buffer: video frames from PhysionDataset.
                    Can be:
                    - list of lists of tensors: [[tensor[C,T,H,W]], ...]
                    - list of tensors: [tensor[C,T,H,W], ...]
                    - tensor: [C, T, H, W]

        Returns:
            torch.Tensor: [T, C, crop_size, crop_size] -- one frame per row,
                          normalized to ImageNet statistics.
        """
        # ---- Unwrap nested list format from PhysionDataset ----
        if isinstance(buffer, list):
            if len(buffer) == 0:
                return torch.empty(0)
            item = buffer[0]
            if isinstance(item, list):
                # [[tensor, ...], ...] → take first clip's first view
                if len(item) > 0:
                    buffer = item[0]
                else:
                    return torch.empty(0)
            else:
                buffer = item

        # Convert numpy → tensor if needed (PhysionDataset returns numpy)
        if isinstance(buffer, np.ndarray):
            # decord returns [T, H, W, C] → permute to [C, T, H, W]
            if buffer.ndim == 4 and buffer.shape[-1] == 3:
                buffer = torch.from_numpy(buffer).permute(3, 0, 1, 2)  # [C,T,H,W]
            elif buffer.ndim == 4 and buffer.shape[1] == 3:
                buffer = torch.from_numpy(buffer)  # already [T,C,H,W] → permute
                buffer = buffer.permute(1, 0, 2, 3)  # [C,T,H,W]
            else:
                buffer = torch.from_numpy(buffer)

        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(buffer)}")

        if buffer.ndim != 4:
            raise ValueError(f"Expected [C, T, H, W], got shape {tuple(buffer.shape)}")

        C, T, H, W = buffer.shape

        # ---- Resize to short_side ----
        scale = self.short_side / min(H, W)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))

        # Permute to [T, C, H, W] for interpolate
        frames = buffer.permute(1, 0, 2, 3)  # [T, C, H, W]
        frames = torch.nn.functional.interpolate(
            frames, size=(new_h, new_w), mode='bilinear', align_corners=False
        )

        # ---- CenterCrop ----
        h_start = (new_h - self.crop_size) // 2
        w_start = (new_w - self.crop_size) // 2
        frames = frames[:, :, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size]

        # ---- Random horizontal flip (training only) ----
        if self.training and self.random_horizontal_flip and torch.rand(1).item() > 0.5:
            frames = frames.flip(-1)

        # ---- Normalize ----
        frames = frames.float() / 255.0
        frames = (frames - self.mean.to(frames.device)) / self.std.to(frames.device)

        return frames  # [T, C, H, W]
