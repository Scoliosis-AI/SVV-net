from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .data import IMAGE_EXTENSIONS


VERTEBRA_TO_ID = {
    **{f"T{level}": level for level in range(1, 13)},
    **{f"L{level}": level + 12 for level in range(1, 6)},
    "T13": 13,
}
NUM_VERTEBRA_CLASSES = 18
DEFAULT_IMAGE_SIZE = (1024, 256)
DEFAULT_ID_COLUMN = "case_id"


def normalize_case_id(value) -> str:
    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.lower().endswith(tuple(IMAGE_EXTENSIONS)):
        return Path(text).stem
    return text


def encode_vertebra(value) -> int:
    if pd.isna(value):
        raise ValueError("UIV/LIV contains an empty value")
    if isinstance(value, (int, float)) and float(value).is_integer():
        encoded = int(value)
        if 1 <= encoded <= 17:
            return encoded
    label = str(value).strip().upper().replace(" ", "")
    if label not in VERTEBRA_TO_ID:
        raise ValueError(
            f"unsupported vertebra label {value!r}; expected T1-T13 or L1-L5"
        )
    return VERTEBRA_TO_ID[label]


def find_images(root: str) -> Dict[str, Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    images = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images:
        raise ValueError(f"no supported images found in: {directory}")
    return images


def load_metadata(
    metadata_path: str,
    id_column: str = DEFAULT_ID_COLUMN,
    uiv_column: str = "UIV",
    liv_column: str = "LIV",
) -> Dict[str, Tuple[int, int]]:
    table = pd.read_excel(metadata_path)
    required = {id_column, uiv_column, liv_column}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"metadata is missing columns: {sorted(missing_columns)}"
        )

    metadata = {}
    for _, row in table.iterrows():
        case_id = normalize_case_id(row[id_column])
        metadata[case_id] = (
            encode_vertebra(row[uiv_column]),
            encode_vertebra(row[liv_column]),
        )
    return metadata


class PairedXrayDataset(Dataset):
    def __init__(
        self,
        preoperative_dir: str,
        postoperative_dir: str,
        metadata_path: str,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        channels: int = 1,
        id_column: str = DEFAULT_ID_COLUMN,
        uiv_column: str = "UIV",
        liv_column: str = "LIV",
    ) -> None:
        preoperative = find_images(preoperative_dir)
        postoperative = find_images(postoperative_dir)
        metadata = load_metadata(
            metadata_path,
            id_column=id_column,
            uiv_column=uiv_column,
            liv_column=liv_column,
        )

        case_ids = sorted(
            set(preoperative) & set(postoperative) & set(metadata)
        )
        if not case_ids:
            raise ValueError(
                "no matched cases across pre-op images, post-op images, "
                "and metadata"
            )
        self.samples: List[Tuple[str, Path, Path, int, int]] = [
            (
                case_id,
                preoperative[case_id],
                postoperative[case_id],
                metadata[case_id][0],
                metadata[case_id][1],
            )
            for case_id in case_ids
        ]
        mode = "L" if channels == 1 else "RGB"
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        case_id, pre_path, post_path, uiv, liv = self.samples[index]
        with Image.open(pre_path) as image:
            preoperative = self.transform(image.convert(self.mode))
        with Image.open(post_path) as image:
            postoperative = self.transform(image.convert(self.mode))
        return {
            "case_id": case_id,
            "preoperative": preoperative,
            "postoperative": postoperative,
            "uiv": torch.tensor(uiv, dtype=torch.long),
            "liv": torch.tensor(liv, dtype=torch.long),
        }
