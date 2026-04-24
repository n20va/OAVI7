from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Лабораторная работа №7. Вариант 11 — Угаритский алфавит
# Классификация символов на основе признаков + сравнение образов
# ============================================================

VARIANT = 11
ALPHABET_NAME = "Угаритский алфавит"
SYMBOLS = [chr(code) for code in range(0x10380, 0x1039E)]  # U+10380..U+1039D, 30 символов

FONT_SIZE = 96
EXPERIMENT_FONT_SIZE = 98

# Каждый символ отделён пробелом, чтобы сегментация была устойчивее.
PHRASE = "𐎀 𐎁 𐎂 𐎍 𐎎 𐎏 𐎛 𐎚 𐎗"
EXPECTED_TEXT = PHRASE.replace(" ", "")

CANVAS_PAD = 30
BIN_THRESHOLD = 200
TRIM_PADDING = 2
NORMALIZED_SIZE = 64

FEATURE_WEIGHT = 0.35
IMAGE_WEIGHT = 0.65

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_lab7"
SRC_DIR = BASE_DIR / "src_lab7"
REPORT_PATH = BASE_DIR / "report_lab7.md"

TEMPLATES_DIR = RESULTS_DIR / "templates"
MAIN_DIR = RESULTS_DIR / "main"
EXP_DIR = RESULTS_DIR / "experiment"

SRC_TEMPLATES_DIR = SRC_DIR / "templates"
SRC_MAIN_DIR = SRC_DIR / "main"
SRC_EXP_DIR = SRC_DIR / "experiment"

FEATURES_CSV = RESULTS_DIR / "alphabet_features.csv"


@dataclass
class Features:
    symbol: str
    unicode_code: str
    mass: float
    xc_norm: float
    yc_norm: float
    ix_norm: float
    iy_norm: float

    def vector(self) -> np.ndarray:
        return np.array(
            [self.mass, self.xc_norm, self.yc_norm, self.ix_norm, self.iy_norm],
            dtype=np.float64,
        )


@dataclass
class Template:
    symbol: str
    image: np.ndarray
    features: Features
    file_name: str


@dataclass
class Segment:
    index: int
    image: np.ndarray
    x0: int
    y0: int
    x1: int
    y1: int
    file_name: str


@dataclass
class RecognitionResult:
    name: str
    font_size: int
    phrase_image: np.ndarray
    segments: list[Segment]
    hypotheses: list[list[tuple[str, float]]]
    predicted: str
    expected: str
    errors: int
    accuracy: float
    folder: Path
    src_folder: Path


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def setup_dirs() -> None:
    for path in [
        RESULTS_DIR,
        SRC_DIR,
        TEMPLATES_DIR,
        MAIN_DIR,
        EXP_DIR,
        SRC_TEMPLATES_DIR,
        SRC_MAIN_DIR,
        SRC_EXP_DIR,
    ]:
        ensure_clean_dir(path)


def find_font_path() -> Path:
    candidates = [
        Path("NotoSansUgaritic-Regular.ttf"),
        BASE_DIR / "NotoSansUgaritic-Regular.ttf",
        Path("/Library/Fonts/NotoSansUgaritic-Regular.ttf"),
        Path.home() / "Library/Fonts/NotoSansUgaritic-Regular.ttf",
        Path("/System/Library/Fonts/Supplemental/NotoSansUgaritic-Regular.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansUgaritic-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansUgaritic-Regular.ttf"),
        Path(r"C:\Windows\Fonts\NotoSansUgaritic-Regular.ttf"),
        Path(r"C:\Windows\Fonts\seguihis.ttf"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Не найден шрифт NotoSansUgaritic-Regular.ttf. "
        "Положите его рядом с lab7_variant11.py или в папку проекта."
    )


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def render_text_mono(
    text: str,
    font: ImageFont.FreeTypeFont,
    pad_x: int = CANVAS_PAD,
    pad_y: int = 12,
) -> np.ndarray:
    canvas_w = 2600
    canvas_h = 360

    image = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (canvas_w - text_w) // 2 - bbox[0]
    y = (canvas_h - text_h) // 2 - bbox[1]

    draw.text((x, y), text, fill=0, font=font)

    arr = np.asarray(image, dtype=np.uint8)
    black = arr < BIN_THRESHOLD
    ys, xs = np.where(black)

    if ys.size == 0:
        raise RuntimeError("Текст не отрисован. Проверьте шрифт с поддержкой угаритского письма.")

    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(arr.shape[0] - 1, int(ys.max()) + pad_y)
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(arr.shape[1] - 1, int(xs.max()) + pad_x)

    cropped = arr[y0 : y1 + 1, x0 : x1 + 1]
    return np.where(cropped < BIN_THRESHOLD, 0, 255).astype(np.uint8)


def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    return render_text_mono(symbol, font, pad_x=TRIM_PADDING, pad_y=TRIM_PADDING)


def compute_features(symbol: str, image: np.ndarray) -> Features:
    mask = (image == 0).astype(np.uint8)

    h, w = mask.shape
    mass_px = int(mask.sum())
    area = max(1, h * w)
    mass = mass_px / area

    y_idx, x_idx = np.where(mask > 0)

    if y_idx.size == 0:
        xc = (w - 1) / 2.0
        yc = (h - 1) / 2.0
        ix = 0.0
        iy = 0.0
    else:
        xc = float(x_idx.mean())
        yc = float(y_idx.mean())
        ix = float(np.sum((y_idx - yc) ** 2))
        iy = float(np.sum((x_idx - xc) ** 2))

    xc_norm = xc / max(1.0, float(w - 1))
    yc_norm = yc / max(1.0, float(h - 1))
    ix_norm = ix / max(1.0, mass_px * (h**2))
    iy_norm = iy / max(1.0, mass_px * (w**2))

    return Features(
        symbol=symbol,
        unicode_code=f"U+{ord(symbol):04X}" if symbol else "SEGMENT",
        mass=mass,
        xc_norm=xc_norm,
        yc_norm=yc_norm,
        ix_norm=ix_norm,
        iy_norm=iy_norm,
    )


def resize_to_square(image: np.ndarray, size: int = NORMALIZED_SIZE) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8)).convert("L")
    pil.thumbnail((size - 8, size - 8), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (size, size), 255)
    x = (size - pil.width) // 2
    y = (size - pil.height) // 2
    canvas.paste(pil, (x, y))

    arr = np.asarray(canvas, dtype=np.uint8)
    return np.where(arr < BIN_THRESHOLD, 0, 255).astype(np.uint8)


def image_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64 = resize_to_square(a)
    b64 = resize_to_square(b)

    ma = (a64 == 0).astype(np.float32)
    mb = (b64 == 0).astype(np.float32)

    diff = float(np.mean(np.abs(ma - mb)))
    return max(0.0, min(1.0, 1.0 - diff))


def normalize_feature_vectors(vectors: list[np.ndarray]) -> list[np.ndarray]:
    data = np.vstack(vectors).astype(np.float64)

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)

    return [(v - mins) / denom for v in vectors]


def feature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dist = float(np.linalg.norm(a - b))
    max_dist = math.sqrt(len(a))

    return max(0.0, min(1.0, 1.0 - dist / max_dist))


def save_profile_plot(profile: np.ndarray, axis_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3), dpi=120)

    x = np.arange(profile.size)
    ax.bar(x, profile, width=0.85, color="black")
    ax.set_title(f"{axis_name}-профиль")
    ax.set_xlabel("Координата")
    ax.set_ylabel("Черные пиксели")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=16))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def extract_segments_by_expected_count(
    mono: np.ndarray,
    expected_count: int,
    output_folder: Path,
    src_folder: Path,
) -> list[Segment]:
    """
    Сегментация строки по вертикальному профилю с принудительным количеством символов.
    Для этой лабораторной известно, что в строке 9 символов без учёта пробелов.
    Алгоритм берёт самые широкие вертикальные пробелы как границы между символами.
    """
    mask = (mono == 0).astype(np.uint8)
    h, w = mask.shape

    v_profile = mask.sum(axis=0)
    cols = np.where(v_profile > 0)[0]

    if cols.size == 0:
        return []

    x_min = int(cols.min())
    x_max = int(cols.max())

    empty_cols = np.where(v_profile[x_min : x_max + 1] == 0)[0] + x_min

    gaps: list[tuple[int, int, int]] = []
    if empty_cols.size > 0:
        start = int(empty_cols[0])
        prev = int(empty_cols[0])

        for value in empty_cols[1:]:
            x = int(value)
            if x == prev + 1:
                prev = x
            else:
                gaps.append((start, prev, prev - start + 1))
                start = x
                prev = x

        gaps.append((start, prev, prev - start + 1))

    needed_gaps = expected_count - 1

    if len(gaps) < needed_gaps:
        raise RuntimeError(
            f"Недостаточно пробелов для сегментации: найдено {len(gaps)}, нужно {needed_gaps}. "
            f"Добавьте пробелы между символами в PHRASE."
        )

    selected_gaps = sorted(gaps, key=lambda item: item[2], reverse=True)[:needed_gaps]
    cuts = sorted([(start + end) // 2 for start, end, _ in selected_gaps])

    bounds = [x_min] + cuts + [x_max + 1]

    segments: list[Segment] = []

    for idx in range(len(bounds) - 1):
        x0 = bounds[idx]
        x1 = bounds[idx + 1] - 1

        part = mask[:, x0 : x1 + 1]
        ys, xs = np.where(part > 0)

        if ys.size == 0:
            continue

        y0 = int(ys.min())
        y1 = int(ys.max())
        real_x0 = x0 + int(xs.min())
        real_x1 = x0 + int(xs.max())

        y0p = max(0, y0 - 1)
        y1p = min(h - 1, y1 + 1)
        x0p = max(0, real_x0 - 1)
        x1p = min(w - 1, real_x1 + 1)

        crop = mono[y0p : y1p + 1, x0p : x1p + 1]
        file_name = f"segment_{idx + 1:02d}.bmp"

        save_gray(crop, output_folder / file_name)
        save_gray(crop, src_folder / file_name)

        segments.append(
            Segment(
                index=idx + 1,
                image=crop,
                x0=x0p,
                y0=y0p,
                x1=x1p,
                y1=y1p,
                file_name=file_name,
            )
        )

    return segments


def draw_boxes(mono: np.ndarray, segments: list[Segment], path: Path) -> None:
    rgb = np.stack([mono, mono, mono], axis=-1)
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    for segment in segments:
        draw.rectangle(
            (segment.x0, segment.y0, segment.x1, segment.y1),
            outline=(255, 0, 0),
            width=2,
        )
        draw.text(
            (segment.x0, max(0, segment.y0 - 14)),
            str(segment.index),
            fill=(255, 0, 0),
        )

    image.save(path)


def build_templates(font: ImageFont.FreeTypeFont) -> list[Template]:
    templates: list[Template] = []
    rows: list[list[str]] = []

    for index, symbol in enumerate(SYMBOLS, start=1):
        code = f"U{ord(symbol):04X}"
        file_name = f"sym_{index:02d}_{code}.bmp"

        image = render_symbol(symbol, font)
        features = compute_features(symbol, image)

        save_gray(image, TEMPLATES_DIR / file_name)
        save_gray(image, SRC_TEMPLATES_DIR / file_name)

        templates.append(
            Template(
                symbol=symbol,
                image=image,
                features=features,
                file_name=file_name,
            )
        )

        rows.append(
            [
                str(index),
                symbol,
                features.unicode_code,
                f"{features.mass:.8f}",
                f"{features.xc_norm:.8f}",
                f"{features.yc_norm:.8f}",
                f"{features.ix_norm:.8f}",
                f"{features.iy_norm:.8f}",
                file_name,
            ]
        )

    with FEATURES_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [
                "index",
                "symbol",
                "unicode",
                "mass",
                "xc_norm",
                "yc_norm",
                "ix_norm",
                "iy_norm",
                "file",
            ]
        )
        writer.writerows(rows)

    return templates


def recognize_segments(
    segments: list[Segment],
    templates: list[Template],
) -> list[list[tuple[str, float]]]:
    raw_vectors = [template.features.vector() for template in templates]
    raw_vectors += [compute_features("", segment.image).vector() for segment in segments]

    normalized = normalize_feature_vectors(raw_vectors)

    template_vectors = normalized[: len(templates)]
    segment_vectors = normalized[len(templates) :]

    all_hypotheses: list[list[tuple[str, float]]] = []

    for segment, segment_vector in zip(segments, segment_vectors):
        hypotheses: list[tuple[str, float]] = []

        for template, template_vector in zip(templates, template_vectors):
            f_sim = feature_similarity(segment_vector, template_vector)
            i_sim = image_similarity(segment.image, template.image)
            score = FEATURE_WEIGHT * f_sim + IMAGE_WEIGHT * i_sim

            hypotheses.append((template.symbol, round(float(score), 6)))

        hypotheses.sort(key=lambda item: item[1], reverse=True)
        all_hypotheses.append(hypotheses)

    return all_hypotheses


def save_hypotheses(hypotheses: list[list[tuple[str, float]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for index, hyp in enumerate(hypotheses, start=1):
            formatted = ", ".join([f"('{symbol}', {score:.6f})" for symbol, score in hyp])
            file.write(f"{index}: [{formatted}]\n")


def evaluate(predicted: str, expected: str) -> tuple[int, float]:
    n = max(len(predicted), len(expected))

    errors = 0
    for index in range(n):
        p = predicted[index] if index < len(predicted) else None
        e = expected[index] if index < len(expected) else None

        if p != e:
            errors += 1

    accuracy = 0.0 if n == 0 else (n - errors) / n * 100.0
    return errors, accuracy


def run_recognition(
    name: str,
    font_path: Path,
    font_size: int,
    templates: list[Template],
    folder: Path,
    src_folder: Path,
) -> RecognitionResult:
    font = ImageFont.truetype(str(font_path), font_size)

    phrase_image = render_text_mono(PHRASE, font)

    save_gray(phrase_image, folder / "phrase_mono.bmp")
    save_gray(phrase_image, src_folder / "phrase_mono.bmp")

    mask = (phrase_image == 0).astype(np.uint8)
    h_profile = mask.sum(axis=1).astype(np.int32)
    v_profile = mask.sum(axis=0).astype(np.int32)

    save_profile_plot(h_profile, "Горизонтальный", folder / "horizontal_profile.png")
    save_profile_plot(v_profile, "Вертикальный", folder / "vertical_profile.png")

    shutil.copy2(folder / "horizontal_profile.png", src_folder / "horizontal_profile.png")
    shutil.copy2(folder / "vertical_profile.png", src_folder / "vertical_profile.png")

    segments = extract_segments_by_expected_count(
        phrase_image,
        expected_count=len(EXPECTED_TEXT),
        output_folder=folder,
        src_folder=src_folder,
    )

    draw_boxes(phrase_image, segments, folder / "segmentation_boxes.png")
    shutil.copy2(folder / "segmentation_boxes.png", src_folder / "segmentation_boxes.png")

    hypotheses = recognize_segments(segments, templates)

    save_hypotheses(hypotheses, folder / f"{name}_hypotheses.txt")

    predicted = "".join(hypothesis[0][0] for hypothesis in hypotheses if hypothesis)
    errors, accuracy = evaluate(predicted, EXPECTED_TEXT)

    with (folder / f"{name}_summary.txt").open("w", encoding="utf-8") as file:
        file.write(f"Размер шрифта: {font_size}\n")
        file.write(f"Файл гипотез: {folder / (name + '_hypotheses.txt')}\n")
        file.write(f"Найдено сегментов: {len(segments)}\n")
        file.write(f"Лучшие гипотезы строкой: {predicted}\n")
        file.write(f"Ожидаемая строка: {EXPECTED_TEXT}\n")
        file.write(f"Ошибок: {errors} из {max(len(predicted), len(EXPECTED_TEXT))}\n")
        file.write(f"Доля верно распознанных символов: {accuracy:.2f}%\n")

    return RecognitionResult(
        name=name,
        font_size=font_size,
        phrase_image=phrase_image,
        segments=segments,
        hypotheses=hypotheses,
        predicted=predicted,
        expected=EXPECTED_TEXT,
        errors=errors,
        accuracy=accuracy,
        folder=folder,
        src_folder=src_folder,
    )


def write_report(
    font_path: Path,
    main_result: RecognitionResult,
    experiment_result: RecognitionResult,
) -> None:
    lines: list[str] = []

    lines.append("# Лабораторная работа №7")
    lines.append("## Классификация на основе признаков, анализ профилей")
    lines.append("")
    lines.append(f"### Вариант {VARIANT}: {ALPHABET_NAME}")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append(f"- Фраза: `{PHRASE}`")
    lines.append(f"- Ожидаемая строка без пробелов: `{EXPECTED_TEXT}`")
    lines.append(f"- Шрифт: `{font_path.name}`")
    lines.append(f"- Основной размер шрифта: `{FONT_SIZE}`")
    lines.append(f"- Размер шрифта в эксперименте: `{EXPERIMENT_FONT_SIZE}`")
    lines.append(f"- Количество символов алфавита: `{len(SYMBOLS)}`")
    lines.append("")
    lines.append("### Метод")
    lines.append("")
    lines.append(
        "Для каждого эталонного символа и каждого сегмента строки рассчитаны признаки: "
        "масса, нормированные координаты центра тяжести и нормированные осевые моменты инерции. "
        "Евклидово расстояние переведено в меру близости. Дополнительно используется сравнение "
        "нормализованных пиксельных образов."
    )
    lines.append("")
    lines.append("```text")
    lines.append("feature_score = 1 - euclidean_distance / sqrt(n)")
    lines.append("image_score = 1 - mean(abs(normalized_segment - normalized_template))")
    lines.append(f"score = {FEATURE_WEIGHT} * feature_score + {IMAGE_WEIGHT} * image_score")
    lines.append("```")
    lines.append("")
    lines.append("Сегментация выполнена по вертикальному профилю с известным числом символов строки.")
    lines.append("")
    lines.append("### Основное распознавание")
    lines.append("")
    lines.append("![main phrase](src_lab7/main/phrase_mono.bmp)")
    lines.append("")
    lines.append("![main boxes](src_lab7/main/segmentation_boxes.png)")
    lines.append("")
    lines.append(f"- Найдено сегментов: `{len(main_result.segments)}`")
    lines.append(f"- Лучшие гипотезы: `{main_result.predicted}`")
    lines.append(f"- Ожидаемая строка: `{main_result.expected}`")
    lines.append(f"- Ошибок: `{main_result.errors}`")
    lines.append(f"- Доля верно распознанных символов: `{main_result.accuracy:.2f}%`")
    lines.append("- Файл гипотез: `results_lab7/main/main_hypotheses.txt`")
    lines.append("")
    lines.append("### Эксперимент с другим размером шрифта")
    lines.append("")
    lines.append("![experiment phrase](src_lab7/experiment/phrase_mono.bmp)")
    lines.append("")
    lines.append("![experiment boxes](src_lab7/experiment/segmentation_boxes.png)")
    lines.append("")
    lines.append(f"- Размер шрифта: `{experiment_result.font_size}`")
    lines.append(f"- Найдено сегментов: `{len(experiment_result.segments)}`")
    lines.append(f"- Лучшие гипотезы: `{experiment_result.predicted}`")
    lines.append(f"- Ожидаемая строка: `{experiment_result.expected}`")
    lines.append(f"- Ошибок: `{experiment_result.errors}`")
    lines.append(f"- Доля верно распознанных символов: `{experiment_result.accuracy:.2f}%`")
    lines.append("- Файл гипотез: `results_lab7/experiment/experiment_hypotheses.txt`")
    lines.append("")
    lines.append("### Примеры первых гипотез")
    lines.append("")
    lines.append("| № сегмента | Топ-5 гипотез основного распознавания |")
    lines.append("|---:|:---|")

    for index, hypotheses in enumerate(main_result.hypotheses, start=1):
        top5 = ", ".join([f"{symbol}: {score:.3f}" for symbol, score in hypotheses[:5]])
        lines.append(f"| {index} | {top5} |")

    lines.append("")
    lines.append("### Вывод")
    lines.append(
        "Реализована классификация символов выбранного алфавита. Для каждого сегмента сформирован "
        "упорядоченный список гипотез, построена строка лучших гипотез, посчитаны ошибки и доля "
        "верно распознанных символов. Проведён эксперимент с изменённым размером шрифта."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_dirs()

    font_path = find_font_path()
    font = ImageFont.truetype(str(font_path), FONT_SIZE)

    templates = build_templates(font)

    main_result = run_recognition(
        name="main",
        font_path=font_path,
        font_size=FONT_SIZE,
        templates=templates,
        folder=MAIN_DIR,
        src_folder=SRC_MAIN_DIR,
    )

    experiment_result = run_recognition(
        name="experiment",
        font_path=font_path,
        font_size=EXPERIMENT_FONT_SIZE,
        templates=templates,
        folder=EXP_DIR,
        src_folder=SRC_EXP_DIR,
    )

    write_report(font_path, main_result, experiment_result)

    print("Лабораторная работа №7 выполнена.")
    print(f"Вариант: {VARIANT} ({ALPHABET_NAME})")
    print(f"Шрифт: {font_path}")

    print("\nОсновное распознавание")
    print(f"Найдено сегментов: {len(main_result.segments)}")
    print(f"Лучшие гипотезы строкой: {main_result.predicted}")
    print(f"Ожидаемая строка: {main_result.expected}")
    print(f"Ошибок: {main_result.errors} из {max(len(main_result.predicted), len(main_result.expected))}")
    print(f"Доля верно распознанных символов: {main_result.accuracy:.2f}%")
    print(f"Файл гипотез: {MAIN_DIR / 'main_hypotheses.txt'}")

    print("\nЭксперимент")
    print(f"Размер шрифта в эксперименте: {EXPERIMENT_FONT_SIZE}")
    print(f"Найдено сегментов: {len(experiment_result.segments)}")
    print(f"Лучшие гипотезы строкой: {experiment_result.predicted}")
    print(f"Ожидаемая строка: {experiment_result.expected}")
    print(f"Ошибок: {experiment_result.errors} из {max(len(experiment_result.predicted), len(experiment_result.expected))}")
    print(f"Доля верно распознанных символов: {experiment_result.accuracy:.2f}%")
    print(f"Файл гипотез: {EXP_DIR / 'experiment_hypotheses.txt'}")

    print(f"\nОтчет: {REPORT_PATH}")
    print(f"Результаты: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
