"""VGGHeads TorchScript adapter for the MIT-licensed head_detector model."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from height_estimation.model_cache import download_model


_REPO_ID = "okupyn/vgg_heads"
_REPO_REVISION = "22672222d2631d8095d01afdf92cc8537e7433cd"
_IMAGE_SIZE = 640


class HeadDetector:
    def __init__(
        self,
        model: str = "vgg_heads_l",
        conf_threshold: float = 0.5,
    ) -> None:
        import torch
        from torchvision.ops import nms

        model_path = download_model(
            _REPO_ID,
            f"{model}.trcd",
            revision=_REPO_REVISION,
            local_filename=f"{model}.trcd",
        )
        self._torch = torch
        self._nms = nms
        self.conf_threshold = conf_threshold
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Any = torch.jit.load(
            str(model_path),
            map_location=self._device,
        )
        self._model.eval()

    def _preprocess(
        self,
        image_rgb: np.ndarray,
    ) -> tuple[Any, tuple[int, int], float]:
        height, width = image_rgb.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be greater than zero")
        if height > width:
            new_height = _IMAGE_SIZE
            new_width = int(width * _IMAGE_SIZE / height)
        else:
            new_height = int(height * _IMAGE_SIZE / width)
            new_width = _IMAGE_SIZE
        scale = _IMAGE_SIZE / max(height, width)
        resized = cv2.resize(
            image_rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        pad_width = _IMAGE_SIZE - resized.shape[1]
        pad_height = _IMAGE_SIZE - resized.shape[0]
        pad_x, pad_y = pad_width // 2, pad_height // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            pad_height - pad_y,
            pad_x,
            pad_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=127,
        )
        tensor = (
            self._torch.from_numpy(padded)
            .to(self._device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        return tensor, (pad_x, pad_y), scale

    def _filter_boxes(
        self,
        boxes: Any,
        scores: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes = boxes.detach().float().squeeze(0)
        scores = scores.detach().float().squeeze(0).squeeze(-1)
        selected = scores >= self.conf_threshold
        boxes, scores = boxes[selected], scores[selected]
        if scores.size(0) > 1000:
            indices = self._torch.topk(
                scores,
                k=1000,
                largest=True,
                sorted=True,
            ).indices
            boxes, scores = boxes[indices], scores[indices]
        if scores.size(0) == 0:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
        indices = self._nms(boxes, scores, 0.5)[:100]
        return boxes[indices].cpu().numpy(), scores[indices].cpu().numpy()

    @staticmethod
    def _restore_boxes(
        boxes: np.ndarray,
        padding: tuple[int, int],
        scale: float,
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes
        restored = boxes.copy().clip(0, _IMAGE_SIZE)
        restored[:, [0, 2]] -= padding[0]
        restored[:, [1, 3]] -= padding[1]
        restored /= scale
        return np.rint(restored).astype(int)

    def detect(self, image: np.ndarray) -> dict[str, object]:
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, padding, scale = self._preprocess(image_rgb)
        with self._torch.no_grad():
            raw_output = self._model(tensor)
        boxes, scores = self._filter_boxes(raw_output[0], raw_output[1])
        boxes = self._restore_boxes(boxes, padding, scale)
        heads = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            heads.append(
                {
                    "bbox": {
                        "x1": float(x1 / width),
                        "y1": float(y1 / height),
                        "x2": float(x2 / width),
                        "y2": float(y2 / height),
                    },
                    "bbox_pixels": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "confidence": float(score),
                    "head_top_y": float(y1 / height),
                }
            )

        result: dict[str, object] = {
            "head_detected": bool(heads),
            "head_top_y": None,
            "head_top_y_pixels": None,
            "head_bbox": None,
            "head_bbox_pixels": None,
            "all_heads": heads,
        }
        if not heads:
            return result

        index = int(
            np.argmax(
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )
        )
        x1, y1, x2, y2 = boxes[index]
        result.update(
            {
                "head_top_y": float(y1 / height),
                "head_top_y_pixels": float(y1),
                "head_bbox": {
                    "x1": float(x1 / width),
                    "y1": float(y1 / height),
                    "x2": float(x2 / width),
                    "y2": float(y2 / height),
                },
                "head_bbox_pixels": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
            }
        )
        return result
