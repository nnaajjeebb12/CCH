import os
import sys
import threading
import pathlib
import subprocess
import logging
from logging.handlers import RotatingFileHandler

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import tensorflow as tf
import numpy as np
import json


class _DummyStream:
    def write(self, _):
        pass

    def flush(self):
        pass


if getattr(sys, "frozen", False):
    if sys.stdout is None:
        sys.stdout = _DummyStream()
    if sys.stderr is None:
        sys.stderr = _DummyStream()


def get_app_dir() -> pathlib.Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return pathlib.Path(sys.executable).resolve().parent
    return pathlib.Path(__file__).resolve().parent


APP_DIR = get_app_dir()
PROJECT_ROOT = APP_DIR

LOG_DIR = PROJECT_ROOT / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAINER_ERROR_LOG = LOG_DIR / "trainer_error.log"
IMAGE_CHECK_LOG = LOG_DIR / "image_check_report.log"

LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUP_COUNT = 10


def setup_rotating_logger(name: str, file_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = RotatingFileHandler(
            file_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


trainer_logger = setup_rotating_logger("trainer_error_logger", TRAINER_ERROR_LOG)
image_check_logger = setup_rotating_logger("image_check_logger", IMAGE_CHECK_LOG)

HUSKTECH_MODEL_DIR = PROJECT_ROOT / "husktech_model"
HUSKTECH_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.widget.bind("<Enter>", self._enter)
        self.widget.bind("<Leave>", self._leave)

    def _enter(self, _event=None):
        self._schedule()

    def _leave(self, _event=None):
        self._unschedule()
        self._hide_tip()

    def _schedule(self):
        self._unschedule()
        self.id = self.widget.after(600, self._show_tip)

    def _unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def _show_tip(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=LEFT,
            background="#ffffe0",
            relief=SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=320,
        )
        label.pack(ipadx=4, ipady=2)

    def _hide_tip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


TRAINER_CONFIG = PROJECT_ROOT / "trainer_config.json"

DEFAULT_DATASET_BASE = PROJECT_ROOT / "dataset"

DATASET_BASE = None
DATA_DIR = None
TRAIN_DIR = None
COLLECTED_SAMPLES_DIR = None
INVALID_IMAGES_DIR = None

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

MODEL_ARCH_JSON = HUSKTECH_MODEL_DIR / "coconut_husk_quality_model_architecture.json"
MODEL_CKPT_DIR = HUSKTECH_MODEL_DIR / "checkpoints"
MODEL_CKPT_PREFIX = MODEL_CKPT_DIR / "ckpt"
MODEL_TFLITE = HUSKTECH_MODEL_DIR / "coconut_husk_quality_model.tflite"
LABELS_TXT = HUSKTECH_MODEL_DIR / "class_names.txt"
MODEL_META_JSON = HUSKTECH_MODEL_DIR / "model_meta.json"
REJECTION_CFG = HUSKTECH_MODEL_DIR / "rejection_config.json"

DEFAULT_MODEL_VERSION = 1
DEFAULT_DRIVE_FOLDER_ID = "1hBEGKnhZN6SSP5sFJhr7qKkvm0VbsUy1"

CLASS_LABELS = ["immature", "mature", "overmature"]

keras = tf.keras
layers = tf.keras.layers


def load_trainer_config():
    if TRAINER_CONFIG.exists():
        try:
            return json.loads(TRAINER_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"dataset_base": str(DEFAULT_DATASET_BASE)}


def save_trainer_config(cfg):
    TRAINER_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def set_dataset_base(path: pathlib.Path, log=None):
    global DATASET_BASE, DATA_DIR, TRAIN_DIR, COLLECTED_SAMPLES_DIR, INVALID_IMAGES_DIR

    DATASET_BASE = path
    DATA_DIR = DATASET_BASE
    TRAIN_DIR = DATA_DIR / "train"
    COLLECTED_SAMPLES_DIR = PROJECT_ROOT / "collected_samples"
    INVALID_IMAGES_DIR = DATASET_BASE / "invalid_images"

    if log:
        log(f"Dataset base set to: {DATASET_BASE}")


_cfg = load_trainer_config()
set_dataset_base(pathlib.Path(_cfg["dataset_base"]))


def ensure_train_subfolders(log):
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for label in CLASS_LABELS:
        folder = TRAIN_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        log(f"Ensured folder: {folder}")


def make_unique_path(dest_path: pathlib.Path) -> pathlib.Path:
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def make_unique_png_path(dest_dir: pathlib.Path, base_name: str) -> pathlib.Path:
    safe_stem = pathlib.Path(base_name).stem
    dest = dest_dir / f"{safe_stem}.png"
    i = 1
    while dest.exists():
        dest = dest_dir / f"{safe_stem}_{i}.png"
        i += 1
    return dest


def validate_training_images(log):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    invalid_files = []

    ensure_train_subfolders(log)
    image_check_logger.info("=== CHECK IMAGES START ===")
    image_check_logger.info("Dataset folder: %s", TRAIN_DIR)

    for label in CLASS_LABELS:
        class_dir = TRAIN_DIR / label
        if not class_dir.exists():
            continue

        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            if ext not in valid_exts:
                invalid_files.append((file_path, f"unsupported extension: {ext or '<none>'}"))
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                invalid_files.append((file_path, f"corrupt/unreadable image: {e}"))

    if invalid_files:
        log("Invalid files found in dataset:")
        image_check_logger.warning("Invalid files found in dataset:")
        for fp, reason in invalid_files:
            line = f"{fp} ({reason})"
            log(f" - {line}")
            image_check_logger.warning(line)
        log(f"Image check report saved to: {IMAGE_CHECK_LOG}")
        image_check_logger.info("Report written to: %s", IMAGE_CHECK_LOG)
    else:
        log("All training images validated successfully.")
        image_check_logger.info("All training images validated successfully.")

    image_check_logger.info("=== CHECK IMAGES END ===")
    return invalid_files


def quarantine_invalid_training_images(log):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    moved_files = []

    ensure_train_subfolders(log)
    INVALID_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_check_logger.info("=== QUARANTINE INVALID IMAGES START ===")
    image_check_logger.info("Train folder: %s", TRAIN_DIR)
    image_check_logger.info("Invalid images folder: %s", INVALID_IMAGES_DIR)

    for label in CLASS_LABELS:
        class_dir = TRAIN_DIR / label
        invalid_class_dir = INVALID_IMAGES_DIR / label
        invalid_class_dir.mkdir(parents=True, exist_ok=True)

        if not class_dir.exists():
            continue

        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue

            reason = None
            ext = file_path.suffix.lower()

            if ext not in valid_exts:
                reason = f"unsupported extension: {ext or '<none>'}"
            else:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    reason = f"corrupt/unreadable image: {e}"

            if reason:
                dest = make_unique_path(invalid_class_dir / file_path.name)
                src = file_path
                file_path.replace(dest)
                moved_files.append((src, dest, reason))
                log(f"Moved invalid file: {src} -> {dest} ({reason})")
                image_check_logger.warning(
                    "Moved invalid file: %s -> %s (%s)", src, dest, reason
                )

    if moved_files:
        log(f"Moved {len(moved_files)} invalid file(s) to: {INVALID_IMAGES_DIR}")
        image_check_logger.info(
            "Moved %d invalid file(s) to %s", len(moved_files), INVALID_IMAGES_DIR
        )
        log(f"Training will continue after skipping {len(moved_files)} invalid file(s).")
    else:
        log("No invalid files needed to be moved.")
        image_check_logger.info("No invalid files needed to be moved.")

    image_check_logger.info("=== QUARANTINE INVALID IMAGES END ===")
    return moved_files


def create_datasets(log, validation_split=0.2, seed=123):
    log(f"Loading dataset from: {TRAIN_DIR}")

    train_ds = keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

    class_names = train_ds.class_names
    log(f"Class names: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def create_raw_train_ds(log):
    log(f"Loading raw dataset from: {TRAIN_DIR}")
    raw_ds = keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    return raw_ds


def create_model(num_classes: int, log):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_augmentation(x)

    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="coconut_husk_maturity_model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    log("Model with data augmentation and normalization created.")
    return model


class GuiTrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log):
        super().__init__()
        self.log_fn = log
        self.total_batches = None

    def on_train_begin(self, logs=None):
        self.log_fn("Training started...")

    def on_epoch_begin(self, epoch, logs=None):
        self.log_fn(f"\nEpoch {epoch + 1} started")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.total_batches is None and hasattr(self, "params"):
            self.total_batches = self.params.get("steps")

        if (batch + 1) % 10 != 0:
            return

        loss = logs.get("loss")
        acc = logs.get("accuracy")

        msg = f"Batch {batch + 1}"
        if self.total_batches:
            msg += f"/{self.total_batches}"
        if loss is not None:
            msg += f" - loss: {loss:.4f}"
        if acc is not None:
            msg += f" - accuracy: {acc:.4f}"

        self.log_fn(msg)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")

        msg = f"Epoch {epoch + 1} completed"
        if loss is not None:
            msg += f" - loss: {loss:.4f}"
        if acc is not None:
            msg += f" - accuracy: {acc:.4f}"
        if val_loss is not None:
            msg += f" - val_loss: {val_loss:.4f}"
        if val_acc is not None:
            msg += f" - val_accuracy: {val_acc:.4f}"

        self.log_fn(msg)

    def on_train_end(self, logs=None):
        self.log_fn("Training finished.")


def save_model_architecture_and_weights(model, log):
    MODEL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_ARCH_JSON, "w", encoding="utf-8") as f:
        f.write(model.to_json())
    log(f"Model architecture saved to {MODEL_ARCH_JSON}")

    model.save_weights(str(MODEL_CKPT_PREFIX))
    log(f"Model checkpoint saved with prefix {MODEL_CKPT_PREFIX}")


def load_saved_model_for_resume(num_classes: int, log):
    if not MODEL_ARCH_JSON.exists():
        raise FileNotFoundError(
            f"Saved model architecture not found:\n{MODEL_ARCH_JSON}"
        )

    ckpt_index = pathlib.Path(str(MODEL_CKPT_PREFIX) + ".index")
    if not ckpt_index.exists():
        raise FileNotFoundError(
            f"Saved model checkpoint not found:\n{ckpt_index}"
        )

    with open(MODEL_ARCH_JSON, "r", encoding="utf-8") as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)

    expected_output_units = model.output_shape[-1]
    if expected_output_units != num_classes:
        raise ValueError(
            f"Saved model output units ({expected_output_units}) do not match "
            f"current dataset classes ({num_classes})."
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.load_weights(str(MODEL_CKPT_PREFIX))
    log(f"Loaded saved architecture from {MODEL_ARCH_JSON}")
    log(f"Loaded saved weights from checkpoint prefix {MODEL_CKPT_PREFIX}")

    return model


def compute_rejection_config(model, raw_train_ds, class_names, log):
    log("Computing rejection config (class centroids and distance stats)...")

    features = {i: [] for i in range(len(class_names))}

    for batch_images, batch_labels in raw_train_ds:
        preds = model(batch_images, training=False).numpy()
        labels = batch_labels.numpy()
        for p, y in zip(preds, labels):
            features[int(y)].append(p)

    centroids = {}
    stats = {}

    for idx, vecs in features.items():
        if not vecs:
            continue
        arr = np.stack(vecs, axis=0)
        c = arr.mean(axis=0)
        dists = np.linalg.norm(arr - c, axis=1)
        centroids[class_names[idx]] = c.tolist()
        stats[class_names[idx]] = {
            "mean_dist": float(dists.mean()),
            "std_dist": float(dists.std()),
        }

    cfg = {
        "centroids": centroids,
        "stats": stats,
    }

    with open(REJECTION_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log(f"Rejection config saved to {REJECTION_CFG}")


def train_and_convert(epochs: int, log, resume: bool = False):
    log(f"TensorFlow version: {tf.__version__}")

    moved_files = quarantine_invalid_training_images(log)
    if moved_files:
        log(
            f"Skipped {len(moved_files)} invalid image(s). See report: {IMAGE_CHECK_LOG}"
        )

    train_ds, val_ds, class_names = create_datasets(log)
    raw_train_ds = create_raw_train_ds(log)

    num_classes = len(class_names)

    if resume:
        log("Loading existing saved model for resume training...")
        model = load_saved_model_for_resume(num_classes=num_classes, log=log)
    else:
        log("Creating a new model for training from scratch...")
        model = create_model(num_classes, log)

    log("\n=== Training coconut husk maturity CNN ===")
    callbacks = [GuiTrainingLogger(log)]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
    )

    save_model_architecture_and_weights(model, log)

    with open(LABELS_TXT, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    log(f"Class names saved to {LABELS_TXT}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    log(f"TFLite model saved to {MODEL_TFLITE}")

    compute_rejection_config(model, raw_train_ds, class_names, log)
    return class_names


def write_model_meta(
    version: int,
    log,
    model_file_id: str = "MODEL_FILE_ID",
    labels_file_id: str = "LABELS_FILE_ID",
    rejection_file_id: str | None = None,
):
    meta = {
        "version": version,
        "model_url": f"https://drive.google.com/uc?export=download&id={model_file_id}",
        "labels_url": f"https://drive.google.com/uc?export=download&id={labels_file_id}",
    }

    if rejection_file_id:
        meta["rejection_url"] = (
            f"https://drive.google.com/uc?export=download&id={rejection_file_id}"
        )

    with open(MODEL_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"model_meta.json saved to {MODEL_META_JSON}")


def ingest_collected_samples(log):
    if not COLLECTED_SAMPLES_DIR.exists():
        log(f"No collected_samples directory at {COLLECTED_SAMPLES_DIR}")
        return

    log(f"Ingesting samples from {COLLECTED_SAMPLES_DIR}...")
    for label_dir in COLLECTED_SAMPLES_DIR.iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        dest_label_dir = TRAIN_DIR / label
        dest_label_dir.mkdir(parents=True, exist_ok=True)

        for img_path in label_dir.glob("*.*"):
            if not img_path.is_file():
                continue

            try:
                dest_path = make_unique_png_path(dest_label_dir, img_path.name)
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.save(dest_path, format="PNG")
                log(f"Copied and converted {img_path} -> {dest_path}")
            except Exception as e:
                log(f"Skipped {img_path}: {e}")

    log("Ingestion complete.")


def auth_drive(log):
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive

    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(str(PROJECT_ROOT / "credentials.json"))
    log("Opening browser for Google Drive authentication...")
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    log("Authenticated with Google Drive.")
    return drive


def upload_file_to_drive(
    drive,
    local_path: pathlib.Path,
    folder_id: str,
    mime_type: str,
    log=None,
) -> str:
    file_name = local_path.name
    gfile = drive.CreateFile({
        "title": file_name,
        "parents": [{"id": folder_id}],
    })
    gfile.SetContentFile(str(local_path))
    gfile["mimeType"] = mime_type
    gfile.Upload()

    file_id = gfile["id"]

    if log:
        log(f"Uploaded {file_name} to Drive with id={file_id}")

    return file_id


def log_exception_to_file(context: str):
    trainer_logger.exception(context)


class HuskTechMenuGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="minty")

        try:
            icon_path = PROJECT_ROOT / "husktech.ico"
            if icon_path.exists():
                self.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not set icon: {e}")

        self.title("HuskTech – Maturity Model Tool")
        self.geometry("1000x660")

        self.epochs_var = tk.IntVar(value=20)
        self.version_var = tk.IntVar(value=1)
        self.folder_id_var = tk.StringVar(value=DEFAULT_DRIVE_FOLDER_ID)

        self.bg_pil = None
        self.bg_photo = None

        self._configure_fonts()
        self._build_widgets()
        self._ensure_dataset_base_exists_or_prompt()

    def _configure_fonts(self):
        import tkinter.font as tkfont

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=11)

        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family="Segoe UI", size=11)

        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(family="Segoe UI", size=13, weight="bold")

        self.title_font = tkfont.Font(family="Segoe UI", size=26, weight="bold")
        self.page_title_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.subtitle_font = tkfont.Font(family="Segoe UI", size=12)

    def _build_widgets(self):
        container = tk.Frame(self, bg="", highlightthickness=0, bd=0)
        container.pack(fill=BOTH, expand=True)

        self.page_frame = tk.Frame(container, bg="", highlightthickness=0, bd=0)
        self.page_frame.pack(fill=BOTH, expand=True)

        console_frame = ttk.Labelframe(container, text="Log", padding=10)
        self.console_frame = console_frame

        self.log_text = tk.Text(console_frame, wrap="word", state="disabled", height=8)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            console_frame, orient="vertical", command=self.log_text.yview
        )
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text["yscrollcommand"] = scrollbar.set

        self._build_main_page_canvas()
        self._build_prepare_page()
        self._build_train_local_page()
        self._build_train_upload_page()

        self.show_page("main")

    def _ensure_dataset_base_exists_or_prompt(self):
        global DATASET_BASE

        if DATASET_BASE is None:
            set_dataset_base(DEFAULT_DATASET_BASE, log=self.log)

        if DATASET_BASE.exists():
            self.log(f"Using existing dataset base: {DATASET_BASE}")
            ensure_train_subfolders(self.log)
            return

        ans = messagebox.askyesno(
            "Dataset location",
            "The dataset folder does not exist yet.\n\n"
            f"Default location:\n{DATASET_BASE}\n\n"
            "Do you want to use this default location?",
        )

        if ans:
            DATASET_BASE.mkdir(parents=True, exist_ok=True)
            ensure_train_subfolders(self.log)
            cfg = load_trainer_config()
            cfg["dataset_base"] = str(DATASET_BASE)
            save_trainer_config(cfg)
            self.log(f"Created dataset base at default: {DATASET_BASE}")
            return

        custom_dir = filedialog.askdirectory(
            title="Select folder for dataset (will contain 'train' etc.)"
        )

        if not custom_dir:
            messagebox.showwarning(
                "Dataset not set",
                "No dataset folder chosen. The application will use the default location "
                "and create it when needed.",
            )
            DATASET_BASE.mkdir(parents=True, exist_ok=True)
            ensure_train_subfolders(self.log)
            cfg = load_trainer_config()
            cfg["dataset_base"] = str(DATASET_BASE)
            save_trainer_config(cfg)
            return

        custom_path = pathlib.Path(custom_dir)
        set_dataset_base(custom_path, log=self.log)
        DATASET_BASE.mkdir(parents=True, exist_ok=True)
        ensure_train_subfolders(self.log)

        cfg = load_trainer_config()
        cfg["dataset_base"] = str(DATASET_BASE)
        save_trainer_config(cfg)

        self.log(f"Created dataset base at chosen location: {DATASET_BASE}")

    def _change_dataset_folder(self):
        global DATASET_BASE

        new_dir = filedialog.askdirectory(
            title="Select new dataset base folder (will contain 'train/immature', etc.)"
        )
        if not new_dir:
            return

        new_path = pathlib.Path(new_dir)
        set_dataset_base(new_path, log=self.log)
        DATASET_BASE.mkdir(parents=True, exist_ok=True)
        ensure_train_subfolders(self.log)

        cfg = load_trainer_config()
        cfg["dataset_base"] = str(DATASET_BASE)
        save_trainer_config(cfg)

        messagebox.showinfo(
            "Dataset folder changed",
            f"Dataset base folder is now:\n{DATASET_BASE}\n\n"
            "Subfolders 'train/immature', 'train/mature', 'train/overmature'\n"
            "have been created if they did not exist.",
        )

    def _build_main_page_canvas(self):
        self.main_canvas = tk.Canvas(self.page_frame, highlightthickness=0, bd=0)
        self.main_canvas.pack(fill=BOTH, expand=True)

        bg_path = PROJECT_ROOT / "main_bg.png"
        if bg_path.exists():
            self.bg_pil = Image.open(bg_path).convert("RGBA")
            print(f"[INFO] Loaded background: {bg_path}, size={self.bg_pil.size}")
        else:
            print(f"[WARNING] Background image not found: {bg_path}")
            self.bg_pil = None

        self.title_text_id = self.main_canvas.create_text(
            0,
            0,
            text="Coconut Husk\nMaturity Model Tool",
            font=self.title_font,
            fill="#444444",
            justify="center",
        )

        self.menu_btn = ttk.Button(
            self.main_canvas, text="Menu", bootstyle=PRIMARY, width=18
        )
        self.prepare_btn = ttk.Button(
            self.main_canvas,
            text="Prepare Model",
            bootstyle="success",
            width=22,
            command=lambda: self.show_page("prepare"),
        )
        self.dataset_btn = ttk.Button(
            self.main_canvas,
            text="Dataset",
            bootstyle="danger",
            width=22,
            command=self._open_dataset_folder,
        )
        self.change_dataset_btn = ttk.Button(
            self.main_canvas,
            text="Change Dataset Folder",
            bootstyle="secondary",
            width=22,
            command=self._change_dataset_folder,
        )
        self.check_images_btn = ttk.Button(
            self.main_canvas,
            text="Check Images",
            bootstyle="warning",
            width=22,
            command=lambda: self._run_in_thread(self._do_check_images),
        )
        self.train_local_btn = ttk.Button(
            self.main_canvas,
            text="Train Model Local",
            bootstyle="info",
            width=22,
            command=lambda: self.show_page("train_local"),
        )
        self.train_upload_btn = ttk.Button(
            self.main_canvas,
            text="Train Model Online",
            bootstyle="success",
            width=22,
            command=lambda: self.show_page("train_upload"),
        )

        self.menu_win = self.main_canvas.create_window(0, 0, window=self.menu_btn)
        self.prepare_win = self.main_canvas.create_window(0, 0, window=self.prepare_btn)
        self.dataset_win = self.main_canvas.create_window(0, 0, window=self.dataset_btn)
        self.change_dataset_win = self.main_canvas.create_window(
            0, 0, window=self.change_dataset_btn
        )
        self.check_images_win = self.main_canvas.create_window(
            0, 0, window=self.check_images_btn
        )
        self.train_local_win = self.main_canvas.create_window(
            0, 0, window=self.train_local_btn
        )
        self.train_upload_win = self.main_canvas.create_window(
            0, 0, window=self.train_upload_btn
        )

        self.main_canvas.bind("<Configure>", self._on_main_canvas_resize)

    def _on_main_canvas_resize(self, event):
        w, h = event.width, event.height

        if self.bg_pil is not None and w > 0 and h > 0:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS

            resized = self.bg_pil.resize((w, h), resample)
            self.bg_photo = ImageTk.PhotoImage(resized)
            self.main_canvas.delete("bg")
            self.main_canvas.create_image(
                0, 0, image=self.bg_photo, anchor="nw", tags="bg"
            )
            self.main_canvas.tag_lower("bg")

        cx = w // 2
        title_y = 85
        self.main_canvas.coords(self.title_text_id, cx, title_y)

        base_y = title_y + 105
        row_gap = 78
        col_gap = 220

        self.main_canvas.coords(self.menu_win, cx, base_y)

        row2_y = base_y + row_gap
        row3_y = base_y + 2 * row_gap

        self.main_canvas.coords(self.prepare_win, cx - col_gap, row2_y)
        self.main_canvas.coords(self.dataset_win, cx, row2_y)
        self.main_canvas.coords(self.change_dataset_win, cx + col_gap, row2_y)

        self.main_canvas.coords(self.check_images_win, cx - col_gap, row3_y)
        self.main_canvas.coords(self.train_local_win, cx, row3_y)
        self.main_canvas.coords(self.train_upload_win, cx + col_gap, row3_y)

    def _build_prepare_page(self):
        self.prepare_page = ttk.Frame(self.page_frame, padding=20)

        header = ttk.Frame(self.prepare_page)
        header.pack(fill=X)

        ttk.Button(
            header,
            text="← Back",
            bootstyle=DANGER,
            command=lambda: self.show_page("main"),
        ).pack(side=LEFT)

        ttk.Label(
            header, text="Prepare Model – Import Images", font=self.page_title_font
        ).pack(side=LEFT, padx=20)

        ttk.Label(
            self.prepare_page,
            text=(
                "Choose images for each maturity level to build or update "
                "the training dataset. Imported files are normalized and saved as PNG."
            ),
            font=self.subtitle_font,
            wraplength=850,
        ).pack(pady=(12, 24))

        btn_frame = ttk.Frame(self.prepare_page)
        btn_frame.pack(pady=10)

        for i, label in enumerate(CLASS_LABELS):
            text = f"Import {label.capitalize()}"
            b = ttk.Button(
                btn_frame,
                text=text,
                bootstyle=PRIMARY,
                width=20,
                command=lambda l=label: self._run_in_thread(
                    lambda: self._do_import_images_for_class(l)
                ),
            )
            b.grid(row=0, column=i, padx=15, pady=10)

    def _build_train_local_page(self):
        self.train_local_page = ttk.Frame(self.page_frame, padding=20)

        header = ttk.Frame(self.train_local_page)
        header.pack(fill=X)

        ttk.Button(
            header,
            text="← Back",
            bootstyle=DANGER,
            command=lambda: self.show_page("main"),
        ).pack(side=LEFT)

        ttk.Label(
            header, text="Train Model Local", font=self.page_title_font
        ).pack(side=LEFT, padx=20)

        ttk.Label(
            self.train_local_page,
            text="Train locally using a new model or resume from an existing saved model.",
            font=self.subtitle_font,
            wraplength=850,
        ).pack(pady=(12, 24))

        form = ttk.Frame(self.train_local_page)
        form.pack(pady=10)

        ttk.Label(form, text="Epochs:").grid(row=0, column=0, sticky=E, padx=8, pady=5)
        epochs_entry = ttk.Entry(form, width=8, textvariable=self.epochs_var)
        epochs_entry.grid(row=0, column=1, sticky=W, padx=8, pady=5)

        ttk.Label(form, text="Model version:").grid(
            row=0, column=2, sticky=E, padx=8, pady=5
        )
        version_entry = ttk.Entry(form, width=8, textvariable=self.version_var)
        version_entry.grid(row=0, column=3, sticky=W, padx=8, pady=5)

        btns = ttk.Frame(self.train_local_page)
        btns.pack(pady=24)

        start_btn = ttk.Button(
            btns,
            text="Start Training Local",
            bootstyle=INFO,
            width=24,
            command=lambda: self._run_in_thread(
                lambda: self._do_train_model_local(resume=False)
            ),
        )
        start_btn.pack(side=LEFT, padx=10)

        resume_btn = ttk.Button(
            btns,
            text="Load / Resume Training Local",
            bootstyle=SUCCESS,
            width=26,
            command=lambda: self._run_in_thread(
                lambda: self._do_train_model_local(resume=True)
            ),
        )
        resume_btn.pack(side=LEFT, padx=10)

        ToolTip(start_btn, "Start Training Local:\nCreate a new model and train from scratch.")
        ToolTip(resume_btn, "Load / Resume Training Local:\nLoad saved architecture + weights and continue training using current dataset, including newly added images.")

    def _build_train_upload_page(self):
        self.train_upload_page = ttk.Frame(self.page_frame, padding=20)

        header = ttk.Frame(self.train_upload_page)
        header.pack(fill=X)

        ttk.Button(
            header,
            text="← Back",
            bootstyle=DANGER,
            command=lambda: self.show_page("main"),
        ).pack(side=LEFT)

        ttk.Label(
            header, text="Train Model Online", font=self.page_title_font
        ).pack(side=LEFT, padx=20)

        ttk.Label(
            self.train_upload_page,
            text=(
                "Train a new model or resume an existing saved model, then upload it "
                "to Google Drive so phones can download it."
            ),
            font=self.subtitle_font,
            wraplength=850,
        ).pack(pady=(12, 24))

        form = ttk.Frame(self.train_upload_page)
        form.pack(pady=10)

        ttk.Label(form, text="Epochs:").grid(row=0, column=0, sticky=E, padx=8, pady=5)
        epochs_entry = ttk.Entry(form, width=8, textvariable=self.epochs_var)
        epochs_entry.grid(row=0, column=1, sticky=W, padx=8, pady=5)

        ttk.Label(form, text="Model version:").grid(
            row=0, column=2, sticky=E, padx=8, pady=5
        )
        version_entry = ttk.Entry(form, width=8, textvariable=self.version_var)
        version_entry.grid(row=0, column=3, sticky=W, padx=8, pady=5)

        ttk.Label(form, text="Drive Folder ID:").grid(
            row=1, column=0, sticky=E, padx=8, pady=5
        )
        folder_entry = ttk.Entry(form, width=45, textvariable=self.folder_id_var)
        folder_entry.grid(row=1, column=1, columnspan=3, sticky=W, padx=8, pady=5)

        btns = ttk.Frame(self.train_upload_page)
        btns.pack(pady=24)

        start_btn = ttk.Button(
            btns,
            text="Start Training Online",
            bootstyle=INFO,
            width=24,
            command=lambda: self._run_in_thread(
                lambda: self._do_train_model_upload_only(resume=False)
            ),
        )
        start_btn.pack(side=LEFT, padx=10)

        resume_btn = ttk.Button(
            btns,
            text="Load / Resume Training Online",
            bootstyle=SUCCESS,
            width=26,
            command=lambda: self._run_in_thread(
                lambda: self._do_train_model_upload_only(resume=True)
            ),
        )
        resume_btn.pack(side=LEFT, padx=10)

        upload_btn = ttk.Button(
            btns,
            text="Upload",
            bootstyle=PRIMARY,
            width=18,
            command=lambda: self._run_in_thread(self._do_upload_only),
        )
        upload_btn.pack(side=LEFT, padx=10)

        ToolTip(start_btn, "Start Training Online:\nCreate a new model and train from scratch.")
        ToolTip(resume_btn, "Load / Resume Training Online:\nLoad saved architecture + weights and continue training using current dataset, including newly added images.")
        ToolTip(upload_btn, "Upload:\nUpload the generated TFLite, labels, rejection config, and metadata to Drive.")

    def log(self, message: str):
        def _append():
            if hasattr(self, "log_text"):
                self.log_text.configure(state="normal")
                self.log_text.insert("end", message + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        self.after(0, _append)
        print(message)

    def _run_in_thread(self, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()

    def show_page(self, name: str):
        for child in self.page_frame.winfo_children():
            child.pack_forget()

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        if name == "main":
            self.console_frame.pack_forget()
            self.main_canvas.pack(fill=BOTH, expand=True)
        else:
            self.console_frame.pack(fill=BOTH, expand=True, padx=10, pady=(10, 0))
            if name == "prepare":
                self.prepare_page.pack(fill=BOTH, expand=True)
            elif name == "train_local":
                self.train_local_page.pack(fill=BOTH, expand=True)
            elif name == "train_upload":
                self.train_upload_page.pack(fill=BOTH, expand=True)

    def _open_dataset_folder(self):
        try:
            ensure_train_subfolders(self.log)
            path = str(TRAIN_DIR)
            self.log(f"Opening dataset folder: {path}")

            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])

        except Exception as e:
            err_msg = str(e)
            self.log(f"ERROR opening dataset folder: {err_msg}")
            self.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Error",
                    f"Could not open dataset folder:\n{msg}",
                ),
            )

    def _do_check_images(self):
        try:
            self.log("=== CHECK IMAGES ===")
            invalid_files = validate_training_images(self.log)

            def on_done():
                if invalid_files:
                    messagebox.showwarning(
                        "Check Images",
                        f"Found {len(invalid_files)} invalid or unsupported file(s).\n\n"
                        f"See the Log panel for details.\n"
                        f"Report file:\n{IMAGE_CHECK_LOG}",
                    )
                else:
                    messagebox.showinfo(
                        "Check Images",
                        f"All training images are valid.\n\nReport file:\n{IMAGE_CHECK_LOG}",
                    )

            self.after(0, on_done)

        except Exception as e:
            err_msg = str(e)
            self.log(f"ERROR checking images: {err_msg}")
            log_exception_to_file("Check Images failed")
            self.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Error",
                    f"Check Images failed:\n{msg}",
                ),
            )

    def _do_import_images_for_class(self, label: str):
        try:
            ensure_train_subfolders(self.log)

            dest_dir = TRAIN_DIR / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            filepaths = filedialog.askopenfilenames(
                title=f"Select images for '{label}'",
                filetypes=[
                    ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"),
                    ("All files", "*.*"),
                ],
            )

            imported_count = 0
            failed_files = []

            for fp in filepaths:
                src = pathlib.Path(fp)
                try:
                    dest = make_unique_png_path(dest_dir, src.name)
                    with Image.open(src) as img:
                        img = img.convert("RGB")
                        img.save(dest, format="PNG")
                    self.log(f"Imported and converted {src} -> {dest}")
                    imported_count += 1
                except Exception as e:
                    failed_files.append((src, str(e)))
                    self.log(f"FAILED to import {src}: {e}")

            if filepaths:
                def on_done():
                    if failed_files:
                        messagebox.showwarning(
                            "Import complete",
                            f"Imported {imported_count} image(s).\n"
                            f"Failed: {len(failed_files)} image(s).\n"
                            "See the log for details.",
                        )
                    else:
                        messagebox.showinfo(
                            "Import complete",
                            f"Imported {imported_count} image(s) into class '{label}'.\n"
                            "All files were converted to PNG.",
                        )

                self.after(0, on_done)

        except Exception as e:
            err_msg = str(e)
            self.log(f"ERROR importing images for {label}: {err_msg}")
            log_exception_to_file(f"Import images for {label} failed")
            self.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Error",
                    f"Import failed:\n{msg}",
                ),
            )

    def _do_train_model_local(self, resume: bool = False):
        try:
            mode = "LOAD / RESUME TRAINING LOCAL" if resume else "START TRAINING LOCAL"
            self.log(f"=== {mode} ===")
            ensure_train_subfolders(self.log)
            ingest_collected_samples(self.log)

            epochs = self.epochs_var.get()
            train_and_convert(epochs=epochs, log=self.log, resume=resume)

            def on_done():
                self.log(
                    "Train Model Local complete. Model saved in 'husktech_model' folder."
                )
                msg_mode = "Resume training finished." if resume else "Training finished."
                messagebox.showinfo(
                    "Train Model Local",
                    f"{msg_mode}\nModel files (architecture.json, checkpoints, .tflite, "
                    "class_names.txt, rejection_config.json)\n"
                    "are saved in the 'husktech_model' folder next to this tool.",
                )

            self.after(0, on_done)

        except Exception as e:
            err_msg = str(e)
            self.log(f"ERROR in Train Model Local: {err_msg}")
            log_exception_to_file("Train Model Local failed")
            self.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Error",
                    f"Train Model Local failed:\n{msg}\n\nDetailed log:\n{TRAINER_ERROR_LOG}",
                ),
            )

    def _do_train_model_upload_only(self, resume: bool = False):
        try:
            mode = "LOAD / RESUME TRAINING ONLINE" if resume else "START TRAINING ONLINE"
            self.log(f"=== {mode} ===")
            ensure_train_subfolders(self.log)

            epochs = self.epochs_var.get()
            train_and_convert(epochs=epochs, log=self.log, resume=resume)

            def on_done():
                self.log(
                    "Training for Online upload complete. Now press 'Upload' to send files to Drive."
                )
                msg_mode = "Resume training finished." if resume else "Training finished."
                messagebox.showinfo(
                    "Train Model Online",
                    f"{msg_mode}\nNow click 'Upload' to send the model to Google Drive.",
                )

            self.after(0, on_done)

        except Exception as e:
            err_msg = str(e)
            self.log(f"ERROR in Train Model Online: {err_msg}")
            log_exception_to_file("Train Model Online failed")
            self.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Error",
                    f"Train Model Online failed:\n{msg}\n\nDetailed log:\n{TRAINER_ERROR_LOG}",
                ),
            )

    def _do_upload_only(self):
        try:
            self.log("=== UPLOAD MODEL ONLY ===")

            folder_id = self.folder_id_var.get().strip()
            if not folder_id:
                raise ValueError("Please set a valid Google Drive Folder ID.")

            drive = auth_drive(self.log)

            if not MODEL_TFLITE.exists() or not LABELS_TXT.exists():
                raise FileNotFoundError(
                    "Required model files not found in 'husktech_model'. "
                    "Please run training first."
                )

            model_id = upload_file_to_drive(
                drive, MODEL_TFLITE, folder_id, "application/octet-stream", self.log
            )

            labels_id = upload_file_to_drive(
                drive, LABELS_TXT, folder_id, "text/plain", self.log
            )

            rejection_id = None
            if REJECTION_CFG.exists():
                rejection_id = upload_file_to_drive(
                    drive, REJECTION_CFG, folder_id, "application/json", self.log
                )

            version = self.version_var.get()
            write_model_meta(
                version=version,
                log=self.log,
                model_file_id=model_id,
                labels_file_id=labels_id,
                rejection_file_id=rejection_id,
            )

            meta_id = upload_file_to_drive(
                drive, MODEL_META_JSON, folder_id, "application/json", self.log
            )

            url = f"https://drive.google.com/uc?export=download&id={meta_id}"
            self.log("\nUse this URL as _metaUrl in Flutter ModelManager:")
            self.log(url)

            def on_done():
                messagebox.showinfo(
                    "Upload Model",
                    f"Upload complete.\nUse this meta URL in Flutter:\n\n{url}",
                )

            self.after(0, on_done)

        except Exception as e:
            err_msg = str(e)
            self.log(f"ERROR in Upload Model: {err_msg}")
            log_exception_to_file("Upload Model failed")
            self.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Error",
                    f"Upload Model failed:\n{msg}\n\nDetailed log:\n{TRAINER_ERROR_LOG}",
                ),
            )


if __name__ == "__main__":
    app = HuskTechMenuGUI()
    app.mainloop()
