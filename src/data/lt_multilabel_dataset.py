from __future__ import annotations

import json
import pickle
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


PathLike = Union[str, Path]


@dataclass(frozen=True)
class LTSample:
    image_path: Path
    target: Tensor
    raw_image_token: str
    positive_indices: Tuple[int, ...]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


def _as_list(value: Any) -> Optional[List[Any]]:
    if isinstance(value, (list, tuple, set)):
        return list(value)

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
        except Exception:
            return None

        if isinstance(converted, list):
            return converted

    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_numeric_vector(value: Any) -> Optional[List[float]]:
    sequence = _as_list(value)

    if sequence is None:
        return None

    if not sequence:
        return []

    if all(_is_number(item) for item in sequence):
        return [float(item) for item in sequence]

    if len(sequence) == 1:
        return _flatten_numeric_vector(sequence[0])

    return None


def _search_numeric_vector(
    value: Any,
    expected_length: int,
) -> Optional[List[float]]:
    vector = _flatten_numeric_vector(value)

    if vector is not None and len(vector) == expected_length:
        return vector

    preferred_keys = (
        "class_freq",
        "class_frequency",
        "frequency",
        "freq",
        "pos_class_freq",
        "positive_class_freq",
        "num_per_class",
        "class_count",
        "class_counts",
    )

    if isinstance(value, dict):
        key_lookup = {
            str(key).lower(): key
            for key in value.keys()
        }

        for preferred_key in preferred_keys:
            if preferred_key in key_lookup:
                original_key = key_lookup[preferred_key]
                result = _search_numeric_vector(
                    value[original_key],
                    expected_length,
                )

                if result is not None:
                    return result

        for nested_value in value.values():
            result = _search_numeric_vector(
                nested_value,
                expected_length,
            )

            if result is not None:
                return result

    sequence = _as_list(value)

    if sequence is not None:
        for nested_value in sequence:
            result = _search_numeric_vector(
                nested_value,
                expected_length,
            )

            if result is not None:
                return result

    return None


def load_class_frequency(
    path: Optional[PathLike],
    num_classes: int,
) -> Optional[Tensor]:
    if path is None:
        return None

    frequency_path = Path(path)

    if not frequency_path.is_file():
        return None

    raw_object = _load_pickle(frequency_path)
    vector = _search_numeric_vector(raw_object, num_classes)

    if vector is None:
        return None

    return torch.tensor(vector, dtype=torch.float32)


def load_label_names(path: PathLike) -> List[str]:
    names: List[str] = []

    with Path(path).open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            tokens = line.split()

            if len(tokens) >= 2 and tokens[0].lstrip("+-").isdigit():
                name = " ".join(tokens[1:])
            elif len(tokens) >= 2 and tokens[-1].lstrip("+-").isdigit():
                name = " ".join(tokens[:-1])
            else:
                name = line

            names.append(name.strip())

    return names


def _parse_split_line(line: str) -> Tuple[str, List[int]]:
    tokens = shlex.split(line, posix=True)

    if not tokens:
        raise ValueError("划分文件中出现空行")

    image_token = tokens[0]
    label_text = " ".join(tokens[1:])

    labels = [
        int(match)
        for match in re.findall(r"[-+]?\d+", label_text)
    ]

    return image_token, labels


def _candidate_image_paths(
    image_token: str,
    image_roots: Sequence[Path],
) -> List[Path]:
    raw_path = Path(image_token)
    candidates: List[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    if raw_path.is_absolute():
        add(raw_path)

    raw_parts = list(raw_path.parts)

    for root in image_roots:
        add(root / raw_path)
        add(root / "JPEGImages" / raw_path)
        add(root / "JPEGImages" / raw_path.name)
        add(root / raw_path.name)

        if raw_parts:
            known_prefixes = {
                root.name,
                "VOC2012",
                "VOCtest",
                "train2017",
                "val2017",
            }

            if raw_parts[0] in known_prefixes and len(raw_parts) > 1:
                stripped = Path(*raw_parts[1:])
                add(root / stripped)

        if raw_path.parent.name == "JPEGImages":
            add(root / "JPEGImages" / raw_path.name)

    expanded: List[Path] = []

    for candidate in candidates:
        if candidate.suffix:
            expanded.append(candidate)
        else:
            expanded.extend(
                [
                    candidate.with_suffix(".jpg"),
                    candidate.with_suffix(".jpeg"),
                    candidate.with_suffix(".png"),
                ]
            )

    unique_paths: List[Path] = []
    seen = set()

    for candidate in expanded:
        key = str(candidate)

        if key not in seen:
            unique_paths.append(candidate)
            seen.add(key)

    return unique_paths


def _resolve_image_path(
    image_token: str,
    image_roots: Sequence[Path],
) -> Path:
    candidates = _candidate_image_paths(
        image_token=image_token,
        image_roots=image_roots,
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    attempted = "\n".join(
        f"  - {candidate}"
        for candidate in candidates[:12]
    )

    raise FileNotFoundError(
        f"无法定位图像：{image_token}\n"
        f"尝试过的路径：\n{attempted}"
    )


def _detect_label_format(
    label_rows: Sequence[Sequence[int]],
    num_classes: int,
) -> str:
    if not label_rows:
        raise ValueError("划分文件没有有效样本")

    is_multi_hot = all(
        len(row) == num_classes
        and set(row).issubset({-1, 0, 1})
        for row in label_rows
    )

    if is_multi_hot:
        return "multi_hot"

    mixed_full_length = any(
        len(row) == num_classes
        for row in label_rows
    )

    if mixed_full_length:
        raise ValueError(
            "标签格式混合：部分行长度等于类别数，"
            "但并非全部为合法 multi-hot"
        )

    return "index_list"


def _count_index_rows(
    label_rows: Sequence[Sequence[int]],
    num_classes: int,
    index_base: int,
) -> Tuple[Tensor, bool]:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    valid = True

    for row in label_rows:
        for raw_index in set(row):
            class_index = raw_index - index_base

            if not 0 <= class_index < num_classes:
                valid = False
                continue

            counts[class_index] += 1.0

    return counts, valid


def _infer_index_base(
    label_rows: Sequence[Sequence[int]],
    num_classes: int,
    reference_frequency: Optional[Tensor],
) -> Tuple[int, str]:
    flat_indices = [
        value
        for row in label_rows
        for value in row
    ]

    if not flat_indices:
        return 0, "所有样本均无正标签，默认使用 0-based"

    if 0 in flat_indices:
        return 0, "检测到标签索引 0"

    if num_classes in flat_indices:
        return 1, f"检测到标签索引 {num_classes}"

    counts_zero, valid_zero = _count_index_rows(
        label_rows,
        num_classes,
        index_base=0,
    )
    counts_one, valid_one = _count_index_rows(
        label_rows,
        num_classes,
        index_base=1,
    )

    if valid_zero and not valid_one:
        return 0, "只有 0-based 解释不存在越界"

    if valid_one and not valid_zero:
        return 1, "只有 1-based 解释不存在越界"

    if reference_frequency is not None:
        zero_distance = torch.abs(
            counts_zero - reference_frequency
        ).sum().item()

        one_distance = torch.abs(
            counts_one - reference_frequency
        ).sum().item()

        if one_distance < zero_distance:
            return (
                1,
                "依据 class_freq.pkl 判断为 1-based，"
                f"L1={one_distance:.4f}",
            )

        return (
            0,
            "依据 class_freq.pkl 判断为 0-based，"
            f"L1={zero_distance:.4f}",
        )

    return 0, "编号方式存在歧义且无频次参考，默认使用 0-based"


def _target_from_multi_hot(
    labels: Sequence[int],
    num_classes: int,
) -> Tensor:
    if len(labels) != num_classes:
        raise ValueError(
            f"multi-hot 长度为 {len(labels)}，"
            f"期望长度为 {num_classes}"
        )

    return torch.tensor(
        [1.0 if value > 0 else 0.0 for value in labels],
        dtype=torch.float32,
    )


def _target_from_indices(
    labels: Sequence[int],
    num_classes: int,
    index_base: int,
) -> Tensor:
    target = torch.zeros(
        num_classes,
        dtype=torch.float32,
    )

    for raw_index in set(labels):
        class_index = raw_index - index_base

        if not 0 <= class_index < num_classes:
            raise IndexError(
                f"标签索引越界：raw={raw_index}, "
                f"index_base={index_base}, "
                f"num_classes={num_classes}"
            )

        target[class_index] = 1.0

    return target


class LTMultiLabelDataset(Dataset):
    """
    通用长尾多标签分类数据集。

    划分文件支持：

    1. 图像路径 + 正标签索引列表
    2. 图像路径 + 长度等于类别数的 multi-hot 标签
    3. 正标签索引可以是 0-based 或 1-based

    默认返回：
        image, target

    return_metadata=True 时返回：
        image, target, metadata
    """

    def __init__(
        self,
        list_file: PathLike,
        image_roots: Sequence[PathLike],
        num_classes: int,
        transform: Optional[Any] = None,
        class_freq_path: Optional[PathLike] = None,
        return_metadata: bool = False,
        verify_images: bool = True,
    ) -> None:
        super().__init__()

        self.list_file = Path(list_file)
        self.image_roots = [
            Path(root)
            for root in image_roots
        ]
        self.num_classes = int(num_classes)
        self.transform = transform
        self.return_metadata = bool(return_metadata)

        if not self.list_file.is_file():
            raise FileNotFoundError(
                f"划分文件不存在：{self.list_file}"
            )

        for root in self.image_roots:
            if not root.is_dir():
                raise FileNotFoundError(
                    f"图像目录不存在：{root}"
                )

        self.reference_frequency = load_class_frequency(
            class_freq_path,
            self.num_classes,
        )

        parsed_rows: List[Tuple[int, str, List[int]]] = []

        with self.list_file.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    image_token, labels = _parse_split_line(line)
                except Exception as exc:
                    raise ValueError(
                        f"解析 {self.list_file} 第 "
                        f"{line_number} 行失败：{exc}"
                    ) from exc

                parsed_rows.append(
                    (line_number, image_token, labels)
                )

        if not parsed_rows:
            raise ValueError(
                f"划分文件没有有效样本：{self.list_file}"
            )

        label_rows = [
            labels
            for _, _, labels in parsed_rows
        ]

        self.label_format = _detect_label_format(
            label_rows,
            self.num_classes,
        )

        if self.label_format == "index_list":
            self.index_base, self.index_base_reason = (
                _infer_index_base(
                    label_rows,
                    self.num_classes,
                    self.reference_frequency,
                )
            )
        else:
            self.index_base = None
            self.index_base_reason = "multi-hot 不使用索引基准"

        samples: List[LTSample] = []

        for line_number, image_token, labels in parsed_rows:
            try:
                if self.label_format == "multi_hot":
                    target = _target_from_multi_hot(
                        labels,
                        self.num_classes,
                    )
                else:
                    assert self.index_base is not None
                    target = _target_from_indices(
                        labels,
                        self.num_classes,
                        self.index_base,
                    )

                if verify_images:
                    image_path = _resolve_image_path(
                        image_token,
                        self.image_roots,
                    )
                else:
                    image_path = Path(image_token)

            except Exception as exc:
                raise RuntimeError(
                    f"构建样本失败：文件={self.list_file}，"
                    f"行号={line_number}，"
                    f"图像={image_token}，"
                    f"标签={labels}，错误={exc}"
                ) from exc

            positive_indices = tuple(
                torch.nonzero(
                    target > 0,
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )

            samples.append(
                LTSample(
                    image_path=image_path,
                    target=target,
                    raw_image_token=image_token,
                    positive_indices=positive_indices,
                )
            )

        self.samples = samples
        self.targets = torch.stack(
            [sample.target for sample in samples],
            dim=0,
        )

        self.class_counts = self.targets.sum(
            dim=0
        ).to(torch.int64)

        if self.reference_frequency is not None:
            rounded_reference = torch.round(
                self.reference_frequency
            ).to(torch.int64)

            self.frequency_matches_reference = torch.equal(
                self.class_counts,
                rounded_reference,
            )
        else:
            self.frequency_matches_reference = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Union[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, Dict[str, Any]],
    ]:
        sample = self.samples[index]

        try:
            with Image.open(sample.image_path) as image_file:
                image = image_file.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"读取图像失败：index={index}, "
                f"path={sample.image_path}, error={exc}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        target = sample.target.clone()

        if not self.return_metadata:
            return image, target

        metadata: Dict[str, Any] = {
            "index": index,
            "image_path": str(sample.image_path),
            "raw_image_token": sample.raw_image_token,
            "positive_indices": sample.positive_indices,
        }

        return image, target, metadata

    def summary(self) -> Dict[str, Any]:
        return {
            "list_file": str(self.list_file),
            "num_samples": len(self),
            "num_classes": self.num_classes,
            "label_format": self.label_format,
            "index_base": self.index_base,
            "index_base_reason": self.index_base_reason,
            "class_counts": self.class_counts.tolist(),
            "frequency_matches_reference": (
                self.frequency_matches_reference
            ),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"samples={len(self)}, "
            f"classes={self.num_classes}, "
            f"format={self.label_format}, "
            f"index_base={self.index_base}, "
            f"list_file='{self.list_file}')"
        )


def build_image_transform(
    image_size: int,
    training: bool,
) -> transforms.Compose:
    """
    多标签图像变换。

    多标签图像中可能存在多个位置分散的目标，因此不使用
    RandomResizedCrop，避免裁掉真实正标签目标却继续保留标签。

    训练阶段：
        1. 完整图像缩放到固定尺寸；
        2. 随机水平翻转；
        3. 轻量 RandAugment；
        4. ImageNet 标准归一化。

    测试阶段：
        仅进行确定性缩放和归一化。
    """

    image_size = int(image_size)

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if training:
        return transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.RandAugment(
                    num_ops=2,
                    magnitude=9,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )



def load_data_config(
    config_path: PathLike,
    dataset_key: str,
) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"数据配置不存在：{config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        all_config = json.load(file)

    if dataset_key not in all_config:
        raise KeyError(
            f"配置中不存在数据集键：{dataset_key}"
        )

    return dict(all_config[dataset_key])


def build_lt_dataset_from_config(
    config: Mapping[str, Any],
    split: str,
    image_size: int,
    return_metadata: bool = False,
    verify_images: bool = True,
) -> LTMultiLabelDataset:
    if split not in {"train", "test"}:
        raise ValueError(
            f"split 必须是 train 或 test，收到：{split}"
        )

    image_roots = [
        config["train_image_root"],
        config["test_image_root"],
    ]

    return LTMultiLabelDataset(
        list_file=config[f"{split}_list"],
        image_roots=image_roots,
        num_classes=int(config["num_classes"]),
        transform=build_image_transform(
            image_size=image_size,
            training=(split == "train"),
        ),
        class_freq_path=(
            config.get("class_freq")
            if split == "train"
            else None
        ),
        return_metadata=return_metadata,
        verify_images=verify_images,
    )
