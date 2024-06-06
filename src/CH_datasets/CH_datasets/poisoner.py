import torch
import numpy as np
from typing import Tuple, Union, List, Dict, Any, Optional
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import ToTensor, ToPILImage

class Poisoner:
    def __init__(
        self,
        p: float,
        classes: Optional[List[int]] = None,
        poison_before_tensor: bool = True,
    ):
        """
        Poisoner class.
        Args:
            p: probability of pasting all artifacts
            classes: list of classes to which the artifacts should be pasted
        """
        self.p = p
        self.classes = classes
        self.poison_before_tensor = poison_before_tensor

    def __call__(self, img: Image, label: int) -> Image:
        """
        Args:
            img: image to be poisoned
            label: label of the image
        Returns:
            poisoned image and label
        """
        if self.classes is not None and label not in self.classes:
            return img
        if np.random.random() > self.p:
            return img
        img = self._poison(img)
        return img

    def _poison(self, img: Image) -> Image:
        """
        Args:
            img: image to be poisoned
        Returns:
            poisoned image
        """
        return img


class PastePoisoner(Poisoner):
    def __init__(
        self,
        artifact_paths: Union[str, List[str]],
        positions: Union[Tuple[int, int], List[Tuple[int, int]]],
        p: float = 1.0,
        opacity: float = 1.0,
        classes: Optional[List[int]] = None,
    ):
        """
        PastePoisoner class.
        Args:
            artifact_paths: list of paths to the artifacts to be pasted
            positions: list of positions where the artifacts should be pasted
            p: probability of pasting all artifacts
            opacity: opacity of the artifacts
            classes: list of classes to which the artifacts should be pasted
        """
        super().__init__(p, classes, poison_before_tensor=True)
        self.artifact_paths = artifact_paths
        if isinstance(artifact_paths, str):
            self.artifact_paths = [artifact_paths]
        self.positions = positions
        if not isinstance(positions[0], Tuple):
            self.positions = [positions]
        self.opacity = opacity
        self.artifacts = []

        for path in self.artifact_paths:
            with Image.open(path) as artifact:
                artifact = artifact.convert("RGBA")
                if self.opacity < 1:
                    red, green, blue, alpha = artifact.split()
                    alpha = alpha.point(lambda p: int(p * self.opacity))
                    artifact = Image.merge("RGBA", (red, green, blue, alpha))
                self.artifacts.append(artifact)

    def _poison(self, img: Image) -> Image:
        """
        Args:
            img: image to be poisoned
        Returns:
            poisoned image
        """
        for artifact, position in zip(self.artifacts, self.positions):
            img = self._paste(img, artifact, position)
        return img

    def _paste(self, img: Image, artifact: Image, position: Tuple[int, int]) -> Image:
        """
        Args:
            img: image to be poisoned
            artifact: artifact to be pasted
            position: position where the artifact should be pasted
        Returns:
            poisoned image
        """
        img = img.convert("RGBA")
        img.paste(artifact, position, artifact)
        return img.convert("RGB")


class PixelPoisoner(Poisoner):
    def __init__(
        self,
        pixels: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        pixel_value: int,
        p: float,
        classes: Optional[List[int]] = None,
    ):
        """
        PixelPoisoner class.
        Args:
            pixels: list of pixel rectangles ((x1,y1),(x2,y2)) to be poisoned

        """
        super().__init__(p, classes, poison_before_tensor=False)
        self.pixels = pixels
        self.pixel_value = pixel_value

    def _poison(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: image to be poisoned
        Returns:
            poisoned image
        """
        for rect in self.pixels:
            img[
                :, rect[0][0] : (rect[0][1] + 1), rect[1][0] : (rect[1][1] + 1)
            ] = self.pixel_value
        return img


class TextPoisoner(Poisoner):
    def __init__(
        self,
        texts: List[str],
        p: float,
        classes: Optional[List[int]] = None,
        font_size: int = 35,
        color: Tuple[int, int, int] = (0, 0, 0),
        position: Tuple[int, int] = (10, 100),
    ):
        """
        Pastes a random text selected from a given list onto the image.
        """
        self.texts = texts
        self.font = ImageFont.truetype("DejaVuSerif-Bold", font_size)
        self.color = color
        self.position = position
        super().__init__(p, classes, poison_before_tensor=False)

    def _poison(self, img: torch.Tensor) -> torch.Tensor:
        img = ToPILImage()(img)

        if len(self.texts) == 1:
            text = self.texts[0]
        else:
            text = self.texts[torch.randint(0, len(self.texts), size=(1,)).item()]
        draw = ImageDraw.Draw(img)
        draw.text(xy=self.position,
                  text=text,
                  fill=self.color,
                  font=self.font)
        img = ToTensor()(img)

        return img
