from pathlib import Path
from typing import Sequence, Tuple, Union

from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int],
        channels: int = 1,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {self.root}")
        self.paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"no supported images found in: {self.root}")

        mode = "L" if channels == 1 else "RGB"
        operations = [transforms.Resize(image_size)]
        if augment:
            operations.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(10 / image_size[1], 10 / image_size[0]),
                )
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels),
            ]
        )
        self.mode = mode
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert(self.mode))


def create_dataloader(
    root: Union[str, Sequence[str]],
    image_size: Tuple[int, int],
    channels: int,
    batch_size: int,
    augment: bool,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    roots = [root] if isinstance(root, str) else list(root)
    datasets = [
        ImageFolderDataset(path, image_size, channels, augment)
        for path in roots
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
