import json
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import requests
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from tensorflow.lite.python.interpreter import Interpreter


# ==========================
#  Constants & Paths
# ==========================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "husktech_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE_NAME = "coconut_husk_quality_model.tflite"
LABELS_FILE_NAME = "class_names.txt"
VERSION_META_NAME = "model_version.json"      # internal local version tracking / fallback
REJECTION_CFG_NAME = "rejection_config.json"
MODEL_META_FILE_NAME = "model_meta.json"      # produced by trainer
CACHE_FILE_NAME = "husktech_cache.bin"

IMG_SIZE = (224, 224)

# Replace this with the real meta URL printed by trainer upload
META_URL = "https://drive.google.com/uc?export=download&id=1hBEGKnhZN6SSP5sFJhr7qKkvm0VbsUy1"


# ==========================
#  Simple binary cache (obfuscated JSON)
# ==========================

_XOR_KEY = b"\x73\x41\x55\x6c\x54\x31\x6b\x79"


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def read_cache(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        dec = _xor_bytes(raw, _XOR_KEY)
        return json.loads(dec.decode("utf-8"))
    except Exception:
        return None


def write_cache(path: pathlib.Path, data: dict):
    try:
        js = json.dumps(data, indent=2).encode("utf-8")
        enc = _xor_bytes(js, _XOR_KEY)
        path.write_bytes(enc)
    except Exception:
        pass


# ==========================
#  ModelManagerPy
# ==========================

class ModelUpdateStatus(Enum):
    UP_TO_DATE = "up_to_date"
    UPDATED = "updated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ModelPathsPy:
    model: pathlib.Path
    labels: pathlib.Path
    version_meta: pathlib.Path


class ModelManagerPy:
    def __init__(self, base_dir: pathlib.Path, meta_url: str, log=print):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.meta_url = meta_url
        self.log = log

        self.paths = ModelPathsPy(
            model=self.base_dir / MODEL_FILE_NAME,
            labels=self.base_dir / LABELS_FILE_NAME,
            version_meta=self.base_dir / VERSION_META_NAME,
        )

    def _log(self, msg: str):
        self.log(msg)

    def has_local_model(self) -> bool:
        return self.paths.model.exists() and self.paths.labels.exists()

    def read_local_model_bytes(self) -> bytes:
        return self.paths.model.read_bytes()

    def read_local_labels(self) -> str:
        return self.paths.labels.read_text(encoding="utf-8")

    def get_current_model_version(self) -> int:
        if not self.paths.version_meta.exists():
            return 0
        try:
            data = json.loads(self.paths.version_meta.read_text(encoding="utf-8"))
            return int(data.get("version", 0))
        except Exception:
            return 0

    def _set_current_model_version(self, version: int) -> None:
        self.paths.version_meta.write_text(
            json.dumps({"version": version}, indent=2),
            encoding="utf-8",
        )

    def check_and_update_model(self, online: bool) -> ModelUpdateStatus:
        if not online:
            self._log("[ModelManagerPy] Offline mode: skipping Drive check.")
            if self.has_local_model():
                self._log(f"[ModelManagerPy] Using local model file: {self.paths.model}")
                self._log(f"[ModelManagerPy] Using local labels file: {self.paths.labels}")
            return ModelUpdateStatus.SKIPPED

        if "YOUR_META_FILE_ID" in self.meta_url or not self.meta_url.strip():
            self._log("[ModelManagerPy] Online mode selected, but META_URL is not configured.")
            self._log("[ModelManagerPy] Falling back to local model files.")
            return ModelUpdateStatus.FAILED

        try:
            self._log("[ModelManagerPy] Downloading meta JSON...")
            resp = requests.get(self.meta_url, timeout=20)
            if resp.status_code != 200:
                self._log(f"[ModelManagerPy] meta download failed: {resp.status_code}")
                return ModelUpdateStatus.FAILED

            self._log(f"[ModelManagerPy] meta JSON downloaded successfully from: {self.meta_url}")
            meta = resp.json()
            new_version = int(meta["version"])
            model_url = meta["model_url"]
            labels_url = meta["labels_url"]

            current_version = self.get_current_model_version()
            if new_version <= current_version:
                self._log(
                    f"[ModelManagerPy] Local model is up to date (v{current_version})."
                )
                self._log(f"[ModelManagerPy] Using existing local model file: {self.paths.model}")
                self._log(f"[ModelManagerPy] Using existing local labels file: {self.paths.labels}")
                return ModelUpdateStatus.UP_TO_DATE

            self._log(f"[ModelManagerPy] Downloading new model version {new_version} ...")

            model_resp = requests.get(model_url, timeout=60)
            labels_resp = requests.get(labels_url, timeout=60)

            if model_resp.status_code != 200 or labels_resp.status_code != 200:
                self._log(
                    f"[ModelManagerPy] download failed: "
                    f"model={model_resp.status_code}, labels={labels_resp.status_code}"
                )
                return ModelUpdateStatus.FAILED

            self.paths.model.write_bytes(model_resp.content)
            self.paths.labels.write_text(labels_resp.text, encoding="utf-8")

            self._log(f"[ModelManagerPy] Model saved to:  {self.paths.model}")
            self._log(f"[ModelManagerPy] Labels saved to: {self.paths.labels}")

            rejection_url = meta.get("rejection_url")
            if rejection_url:
                try:
                    rej_resp = requests.get(rejection_url, timeout=60)
                    if rej_resp.status_code == 200:
                        rej_path = self.base_dir / REJECTION_CFG_NAME
                        rej_path.write_bytes(rej_resp.content)
                        self._log(f"[ModelManagerPy] rejection config saved to: {rej_path}")
                    else:
                        self._log(
                            f"[ModelManagerPy] rejection config download failed: {rej_resp.status_code}"
                        )
                except Exception as e:
                    self._log(f"[ModelManagerPy] rejection config update error: {e}")

            meta_path = self.base_dir / MODEL_META_FILE_NAME
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._log(f"[ModelManagerPy] model_meta.json saved to: {meta_path}")

            self._set_current_model_version(new_version)
            self._log(f"[ModelManagerPy] Updated model to version {new_version}")
            return ModelUpdateStatus.UPDATED
        except Exception as e:
            self._log(f"[ModelManagerPy] Error: {e}")
            return ModelUpdateStatus.FAILED


# ==========================
#  Inference Client
# ==========================

class HuskTechPythonClient:
    def __init__(self, manager: ModelManagerPy, log=print):
        self.manager = manager
        self.log = log

        self.interpreter: Optional[Interpreter] = None
        self.labels: List[str] = []
        self.model_version: int = 0

        self.centroids = {}
        self.dist_stats = {}

        self.conf_threshold: float = 0.7
        self.margin_threshold: float = 0.2

        self.cache_path = MODEL_DIR / CACHE_FILE_NAME
        self.cached_model_version: int = 0
        self.current_model_version_meta: int = 0

        self._load_cache_and_meta()

    def _read_model_meta_version(self) -> int:
        meta_path = MODEL_DIR / MODEL_META_FILE_NAME
        if not meta_path.exists():
            return 0
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta.get("version", 0))
        except Exception as e:
            self.log(f"[Client] Failed to read version from model_meta.json: {e}")
            return 0

    def _sync_local_version_from_meta(self):
        """
        If model_meta.json has a valid version, mirror it into model_version.json.
        """
        version = self._read_model_meta_version()
        if version > 0:
            try:
                self.manager._set_current_model_version(version)
                self.log(f"[Client] Synced local model_version.json from model_meta.json: version={version}")
            except Exception as e:
                self.log(f"[Client] Failed to sync model_version.json from meta: {e}")

    def _load_cache_and_meta(self):
        cache = read_cache(self.cache_path)
        if cache:
            try:
                self.cached_model_version = int(cache.get("version", 0))
                conf = float(cache.get("conf_threshold", self.conf_threshold))
                margin = float(cache.get("margin_threshold", self.margin_threshold))
                if 0.0 <= conf <= 1.0:
                    self.conf_threshold = conf
                if 0.0 <= margin <= 1.0:
                    self.margin_threshold = margin
                self.log(
                    f"[Client] Loaded cache: version={self.cached_model_version}, "
                    f"CONF_THRESHOLD={self.conf_threshold:.3f}, "
                    f"MARGIN_THRESHOLD={self.margin_threshold:.3f}"
                )
            except Exception as e:
                self.log(f"[Client] Invalid cache file; ignoring: {e}")
        else:
            self.log("[Client] No cache file; using default thresholds.")

        self.current_model_version_meta = self._read_model_meta_version()
        self.log(f"[Client] model_meta.json version={self.current_model_version_meta}")

        # PATCH: auto-sync local version fallback if meta exists
        self._sync_local_version_from_meta()

    def _save_cache(self):
        data = {
            "version": self.current_model_version_meta,
            "conf_threshold": self.conf_threshold,
            "margin_threshold": self.margin_threshold,
        }
        write_cache(self.cache_path, data)
        self.cached_model_version = self.current_model_version_meta
        self.log(f"[Client] Saved cache: {data}")

    def _load_rejection_config(self):
        cfg_path = MODEL_DIR / REJECTION_CFG_NAME
        if not cfg_path.exists():
            self.log("[Client] No rejection_config.json found; distance-based rejection disabled.")
            return
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.centroids = {
                cls: np.array(vec, dtype=np.float32)
                for cls, vec in data.get("centroids", {}).items()
            }
            self.dist_stats = data.get("stats", {})
            self.log(f"[Client] Loaded rejection config from: {cfg_path}")
            self.log(f"[Client] Centroids for classes: {list(self.centroids.keys())}")
        except Exception as e:
            self.log(f"[Client] Failed to load rejection config: {e}")

    def _resolve_current_version(self) -> int:
        """
        Resolve current version with trainer-aligned priority:
        1. model_meta.json
        2. model_version.json
        3. 0
        """
        version = self._read_model_meta_version()
        if version > 0:
            self.manager._set_current_model_version(version)
            return version

        return self.manager.get_current_model_version()

    def load_model_and_labels(self, online: bool):
        if online:
            self.log("=== Loading / Updating Model (Online mode) ===")
        else:
            self.log("=== Loading Model (Offline mode) ===")

        status = self.manager.check_and_update_model(online=online)

        if self.manager.has_local_model():
            model_bytes = self.manager.read_local_model_bytes()
            labels_str = self.manager.read_local_labels()

            self.log(f"[Client] Loaded model bytes from:  {self.manager.paths.model}")
            self.log(f"[Client] Loaded labels text from: {self.manager.paths.labels}")

            self.current_model_version_meta = self._resolve_current_version()
            self.log(f"[Client] Current model version = {self.current_model_version_meta}")

            self._init_interpreter_and_labels(model_bytes, labels_str)
            self._load_rejection_config()
            self._save_cache()

            if online:
                if status == ModelUpdateStatus.UPDATED:
                    self.log("[Client] Online update completed successfully. Downloaded, saved, and validated model from Drive.")
                elif status == ModelUpdateStatus.UP_TO_DATE:
                    self.log("[Client] Online check completed successfully. Local model already up to date.")
                elif status == ModelUpdateStatus.FAILED:
                    self.log("[Client] Online update failed, but local model is still available and was loaded.")
            else:
                self.log("[Client] Offline load completed successfully using local files.")
        else:
            raise FileNotFoundError("no_local_model")

    def load_model_and_labels_from_paths(self, model_path: pathlib.Path, labels_path: pathlib.Path):
        self.log(
            f"[Client] Loading model from user-selected files:\n"
            f"  Model:  {model_path}\n"
            f"  Labels: {labels_path}"
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dest_model = self.manager.paths.model
        dest_labels = self.manager.paths.labels

        dest_model.write_bytes(model_path.read_bytes())
        dest_labels.write_text(labels_path.read_text(encoding="utf-8"), encoding="utf-8")

        self.log(f"[Client] Saved copied model to:  {dest_model}")
        self.log(f"[Client] Saved copied labels to: {dest_labels}")

        src_rejection = model_path.with_name(REJECTION_CFG_NAME)
        dest_rejection = MODEL_DIR / REJECTION_CFG_NAME
        if src_rejection.exists():
            try:
                dest_rejection.write_bytes(src_rejection.read_bytes())
                self.log(f"[Client] Copied rejection config to: {dest_rejection}")
            except Exception as e:
                self.log(f"[Client] Failed to copy rejection config: {e}")

        src_meta = model_path.with_name(MODEL_META_FILE_NAME)
        dest_meta = MODEL_DIR / MODEL_META_FILE_NAME
        version = 0
        if src_meta.exists():
            try:
                meta_text = src_meta.read_text(encoding="utf-8")
                dest_meta.write_text(meta_text, encoding="utf-8")
                meta = json.loads(meta_text)
                version = int(meta.get("version", 0))
                self.log(f"[Client] Copied model_meta.json to: {dest_meta}")
                self.log(f"[Client] Loaded model version {version} from copied model_meta.json")
            except Exception as e:
                self.log(f"[Client] Failed to read/copy {src_meta}: {e}")
        else:
            self.log(f"[Client] No {MODEL_META_FILE_NAME} found next to model; version set to 0.")

        self.current_model_version_meta = version
        if version > 0:
            self.manager._set_current_model_version(version)
        else:
            # PATCH: preserve fallback sync behavior
            self._sync_local_version_from_meta()

        model_bytes = dest_model.read_bytes()
        labels_str = dest_labels.read_text(encoding="utf-8")

        self._init_interpreter_and_labels(model_bytes, labels_str)
        self._load_rejection_config()
        self._save_cache()

        self.log("[Client] Local manually-selected model loaded and validated successfully.")

    def _init_interpreter_and_labels(self, model_bytes: bytes, labels_str: str):
        self.interpreter = Interpreter(model_content=model_bytes)
        self.interpreter.allocate_tensors()

        self.labels = [
            line.strip()
            for line in labels_str.splitlines()
            if line.strip()
        ]
        self.model_version = self.current_model_version_meta

        self.log("[Client] Model initialized.")
        self.log(f"  - Version: {self.model_version}")
        self.log(f"  - Classes: {self.labels}")
        self.log(f"  - CONF_THRESHOLD: {self.conf_threshold}")
        self.log(f"  - MARGIN_THRESHOLD: {self.margin_threshold}")

    def preprocess_image(self, image_path: pathlib.Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        arr = arr.astype("float32")
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image_path: pathlib.Path):
        if self.interpreter is None:
            raise RuntimeError("Model not loaded.")

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

        x = self.preprocess_image(image_path)

        if input_details[0]["dtype"] == np.float32:
            x = x.astype(np.float32)
        elif input_details[0]["dtype"] == np.uint8:
            x = x.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported input dtype: {input_details[0]['dtype']}")

        self.interpreter.set_tensor(input_index, x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_index)

        probs = output[0].astype(float).tolist()
        if not probs:
            raise RuntimeError("Empty prediction output.")

        probs_arr = np.array(probs, dtype=np.float32)
        max_idx = int(np.argmax(probs_arr))
        max_score = float(probs_arr[max_idx])

        if self.labels and 0 <= max_idx < len(self.labels):
            predicted_class = self.labels[max_idx]
        else:
            predicted_class = "unknown"

        rejected = False

        if max_score < self.conf_threshold:
            self.log(
                f"[Client] Rejecting due to low confidence: "
                f"{max_score:.4f} < {self.conf_threshold:.2f}"
            )
            rejected = True

        if not rejected and len(probs_arr) >= 2:
            sorted_probs = np.sort(probs_arr)[::-1]
            top1 = float(sorted_probs[0])
            top2 = float(sorted_probs[1])
            margin = top1 - top2
            if margin < self.margin_threshold:
                self.log(
                    f"[Client] Rejecting due to small margin: "
                    f"top1-top2 = {margin:.4f} < {self.margin_threshold:.2f}"
                )
                rejected = True

        if rejected:
            predicted_class = "rejected"

        return predicted_class, max_score, probs

    @staticmethod
    def explanation_for_class(predicted_class: str) -> str:
        if predicted_class == "mature":
            return "Mature husk: ideal fiber development and moisture level, suitable for processing."
        elif predicted_class == "immature":
            return "Immature husk: fibers not fully developed, higher moisture, may be unsuitable for some uses."
        elif predicted_class == "overmature":
            return "Overmature husk: fibers may be too dry or brittle, possible surface degradation."
        elif predicted_class == "rejected":
            return (
                "Rejected: the image does not sufficiently match any known husk maturity "
                "class. Please capture a clear coconut husk image on a plain background."
            )
        else:
            return "Unknown maturity class."


# ==========================
#  GUI
# ==========================

class HuskTechAppGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="minty")

        self.title("HuskTech – Coconut Husk Maturity")
        self.geometry("900x600")

        self.online_mode = tk.BooleanVar(value=False)

        self.manager = ModelManagerPy(MODEL_DIR, META_URL, log=self.log)
        self.client = HuskTechPythonClient(self.manager, log=self.log)

        self.current_image_tk = None
        self.current_image_path: Optional[pathlib.Path] = None

        self.pred_label_var = tk.StringVar(value="N/A")

        self.log_visible = True

        self.load_click_count = 0
        self.last_load_click_ts: Optional[float] = None

        self._build_layout()
        self._handle_startup_model()

    def _build_layout(self):
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill=BOTH, expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 12))

        ttk.Label(
            header_frame,
            text="Coconut Husk Maturity Classification",
            font=("Segoe UI", 18, "bold"),
        ).pack(side=LEFT)

        self.model_info_var = tk.StringVar(value="Model version: N/A")
        ttk.Label(
            header_frame,
            textvariable=self.model_info_var,
            bootstyle=SECONDARY,
        ).pack(side=RIGHT, padx=8)

        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(fill=X, pady=(0, 8))
        ttk.Label(mode_frame, text="Model mode: ").pack(side=LEFT)

        self.mode_switch = ttk.Checkbutton(
            mode_frame,
            text="Online (Drive)",
            bootstyle="success-round-toggle",
            variable=self.online_mode,
            onvalue=True,
            offvalue=False,
            command=self.on_mode_changed,
        )
        self.mode_switch.pack(side=LEFT, padx=4)

        ttk.Label(
            mode_frame,
            text="(Off = Offline (Local Only / Browse), On = Online (Auto-Update from Drive))",
            bootstyle=SECONDARY,
        ).pack(side=LEFT, padx=12)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=X, pady=(8, 4))

        self.load_button = ttk.Button(
            btn_frame,
            text="Load/Update Model",
            bootstyle=PRIMARY,
            command=self.on_load_model_clicked,
            width=20,
        )
        self.load_button.pack(side=LEFT, padx=4)

        ttk.Button(
            btn_frame,
            text="Select Image & Analyze",
            bootstyle=SUCCESS,
            command=self.on_select_and_analyze,
            width=22,
        ).pack(side=LEFT, padx=4)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=BOTH, expand=True)

        self.image_panel = ttk.Labelframe(content_frame, text="Image", padding=8)
        self.image_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))

        self.image_label = ttk.Label(
            self.image_panel,
            text="No image selected.\nClick 'Select Image & Analyze'.",
            anchor=CENTER,
            justify=CENTER,
        )
        self.image_label.pack(fill=BOTH, expand=True)

        self.result_panel = ttk.Labelframe(content_frame, text="Result", padding=8)
        self.result_panel.pack(side=RIGHT, fill=BOTH, expand=True, padx=(6, 0))

        result_header = ttk.Frame(self.result_panel)
        result_header.pack(fill=X, pady=(0, 4))

        self.pred_label_title = ttk.Label(
            result_header,
            text="Predictive Maturity:",
            font=("Segoe UI", 11, "bold"),
        )
        self.pred_label_title.pack(side=LEFT)

        self.pred_label = ttk.Label(
            result_header,
            textvariable=self.pred_label_var,
            font=("Segoe UI", 11, "bold"),
            bootstyle=INFO,
        )
        self.pred_label.pack(side=LEFT, padx=(6, 0))

        self.result_text = tk.Text(self.result_panel, wrap="word", height=10)
        self.result_text.pack(fill=BOTH, expand=True)

        log_header_frame = ttk.Frame(main_frame)
        log_header_frame.pack(fill=X, pady=(8, 0))

        ttk.Label(
            log_header_frame,
            text="Log",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=LEFT)

        self.toggle_log_btn = ttk.Button(
            log_header_frame,
            text="Hide Log",
            bootstyle=SECONDARY,
            width=10,
            command=self._toggle_log_panel,
        )
        self.toggle_log_btn.pack(side=RIGHT)

        self.log_frame = ttk.Frame(main_frame)
        self.log_frame.pack(fill=BOTH, expand=False, pady=(2, 0))

        log_labelframe = ttk.Labelframe(self.log_frame, padding=8, text="")
        log_labelframe.pack(fill=BOTH, expand=True)

        self.log_text = tk.Text(log_labelframe, wrap="word", height=6, state="disabled")
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        log_scroll = ttk.Scrollbar(
            log_labelframe, orient="vertical", command=self.log_text.yview
        )
        log_scroll.pack(side=RIGHT, fill=Y)
        self.log_text["yscrollcommand"] = log_scroll.set

    def log(self, message: str):
        if hasattr(self, "log_text"):
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        print(message)

    def _toggle_log_panel(self):
        if self.log_visible:
            self.log_frame.forget()
            self.toggle_log_btn.configure(text="Show Log")
            self.log_visible = False
        else:
            self.log_frame.pack(fill=BOTH, expand=False, pady=(2, 0))
            self.toggle_log_btn.configure(text="Hide Log")
            self.log_visible = True

    def _set_header_model_version(self, version: int):
        if version > 0:
            self.model_info_var.set(f"Model version: {version}")
        else:
            self.model_info_var.set("Model version: N/A")

    def _handle_startup_model(self):
        if not self.manager.has_local_model():
            self.log("[GUI] No local model on startup. Please load a model.")
            return

        meta_version = self.client.current_model_version_meta
        cached_version = self.client.cached_model_version

        self.log(
            f"[GUI] Startup: meta_version={meta_version}, cached_version={cached_version}"
        )

        if meta_version > cached_version:
            resp = messagebox.askyesno(
                "New model version detected",
                f"A newer model version ({meta_version}) is available.\n\n"
                f"Previously loaded version: {cached_version or 'none'}.\n\n"
                "Do you want to load the new model now?",
            )
            if resp:
                try:
                    self.client.load_model_and_labels(online=False)
                    self._set_header_model_version(self.client.model_version)
                except Exception as e:
                    self.log(f"[GUI] Auto-load new model failed: {e}")
                    messagebox.showerror("Error", f"Failed to load new model:\n{e}")
            else:
                self.log("[GUI] User chose not to load new model on startup.")
        elif meta_version == cached_version and meta_version > 0:
            try:
                self.log("[GUI] Auto-loading cached model on startup...")
                self.client.load_model_and_labels(online=False)
                self._set_header_model_version(self.client.model_version)
            except Exception as e:
                self.log(f"[GUI] Auto-load cached model failed: {e}")
        else:
            # PATCH: if local version file exists, still reflect it
            fallback_version = self.manager.get_current_model_version()
            self._set_header_model_version(fallback_version)
            self.log("[GUI] Model version unknown from meta; using fallback local version if available.")

    def on_mode_changed(self):
        mode = "Online (Drive)" if self.online_mode.get() else "Offline (Local/Browse)"
        self.log(f"[GUI] Model mode changed to: {mode}")

    def on_load_model_clicked(self):
        import time
        now = time.time()

        if self.last_load_click_ts is None or (now - self.last_load_click_ts) > 10.0:
            self.load_click_count = 0

        self.load_click_count += 1
        self.last_load_click_ts = now

        if self.load_click_count >= 5:
            self.load_click_count = 0
            self._show_thresholds_dialog()
        else:
            self.on_load_model()

    def _show_thresholds_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Advanced thresholds")
        dlg.transient(self)
        dlg.grab_set()

        help_text = (
            "These thresholds control when an image is marked as REJECTED.\n\n"
            "Min top-class probability:\n"
            "  - Range: 0–1 (e.g., 0.70 = 70%).\n"
            "  - Higher value = stricter.\n\n"
            "Min margin (top1 - top2):\n"
            "  - Range: 0–1 (e.g., 0.20 = 20% gap).\n"
            "  - Higher value = stricter.\n\n"
            "Suggested starting values:\n"
            "  - Min top-class probability: 0.70\n"
            "  - Min margin (top1 - top2): 0.20–0.25\n"
        )

        ttk.Label(
            dlg,
            text=help_text,
            bootstyle=SECONDARY,
            wraplength=380,
            justify=LEFT,
        ).pack(padx=12, pady=(10, 12))

        frm = ttk.Frame(dlg)
        frm.pack(padx=12, pady=6, fill=X)

        ttk.Label(frm, text="Min top-class probability:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        conf_var = tk.StringVar(value=f"{self.client.conf_threshold:.2f}")
        conf_entry = ttk.Entry(frm, textvariable=conf_var, width=8)
        conf_entry.grid(row=0, column=1, sticky=W, padx=4, pady=4)

        ttk.Label(frm, text="Min margin (top1 - top2):").grid(row=1, column=0, sticky=E, padx=4, pady=4)
        margin_var = tk.StringVar(value=f"{self.client.margin_threshold:.2f}")
        margin_entry = ttk.Entry(frm, textvariable=margin_var, width=8)
        margin_entry.grid(row=1, column=1, sticky=W, padx=4, pady=4)

        def on_save():
            try:
                new_conf = float(conf_var.get())
                new_margin = float(margin_var.get())
                if not (0.0 <= new_conf <= 1.0 and 0.0 <= new_margin <= 1.0):
                    raise ValueError("Thresholds must be in [0, 1].")

                self.client.conf_threshold = new_conf
                self.client.margin_threshold = new_margin
                self.client._save_cache()
                self.log(
                    f"[GUI] Updated thresholds: CONF_THRESHOLD={new_conf:.3f}, "
                    f"MARGIN_THRESHOLD={new_margin:.3f}"
                )
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Invalid values", f"Failed to update thresholds:\n{e}")

        btns = ttk.Frame(dlg)
        btns.pack(pady=(8, 10))

        ttk.Button(btns, text="OK", bootstyle=PRIMARY, command=on_save, width=10).pack(side=LEFT, padx=4)
        ttk.Button(btns, text="Cancel", bootstyle=SECONDARY, command=dlg.destroy, width=10).pack(side=LEFT, padx=4)

        dlg.geometry("+%d+%d" % (self.winfo_rootx() + 120, self.winfo_rooty() + 120))

    def _prompt_for_local_model_files(self):
        messagebox.showinfo(
            "Select Model Files",
            "No local model found.\n\n"
            "Please select the TFLite model file (coconut_husk_quality_model.tflite).",
        )
        model_path_str = filedialog.askopenfilename(
            title="Select TFLite model file",
            filetypes=[("TFLite model", "*.tflite"), ("All files", "*.*")],
        )
        if not model_path_str:
            raise FileNotFoundError("Model selection cancelled by user.")

        model_path = pathlib.Path(model_path_str)

        messagebox.showinfo(
            "Select Labels File",
            "Now select the labels file (class_names.txt).",
        )
        labels_path_str = filedialog.askopenfilename(
            title="Select labels file",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
        )
        if not labels_path_str:
            raise FileNotFoundError("Labels selection cancelled by user.")

        labels_path = pathlib.Path(labels_path_str)

        self.client.load_model_and_labels_from_paths(model_path, labels_path)
        self._set_header_model_version(self.client.model_version)

    def on_load_model(self):
        try:
            online = self.online_mode.get()
            self.client.load_model_and_labels(online=online)
            self._set_header_model_version(self.client.model_version)
        except FileNotFoundError as e:
            self.log(f"[GUI] Model not found via manager: {e}")
            if str(e) == "no_local_model":
                try:
                    self._prompt_for_local_model_files()
                except FileNotFoundError as browse_err:
                    self.log(f"[GUI] User cancelled model selection: {browse_err}")
                except Exception as ex:
                    self.log(f"[GUI] ERROR selecting model files: {ex}")
                    messagebox.showerror("Error", f"Failed to load model from files:\n{ex}")
            else:
                messagebox.showerror("Model not found", f"{e}")
        except Exception as e:
            self.log(f"[GUI] ERROR loading model: {e}")
            messagebox.showerror("Error", f"Failed to load/update model:\n{e}")

    def on_select_and_analyze(self):
        if self.client.interpreter is None and self.manager.has_local_model():
            try:
                self.log("[GUI] Interpreter is None; re-loading local model...")
                self.client.load_model_and_labels(online=False)
                self._set_header_model_version(self.client.model_version)
            except Exception as e:
                self.log(f"[GUI] ERROR re-loading model before analysis: {e}")

        if self.client.interpreter is None:
            messagebox.showwarning(
                "Model not loaded",
                "Please click 'Load/Update Model' first or restart after selecting a model.",
            )
            return

        filepaths = filedialog.askopenfilenames(
            title="Select coconut husk image",
            filetypes=[
                ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"),
                ("All files", "*.*"),
            ],
        )
        if not filepaths:
            return

        image_path = pathlib.Path(filepaths[0])
        self.current_image_path = image_path
        self.log(f"[GUI] Selected image: {image_path}")

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((320, 240), Image.BILINEAR)
            self.current_image_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.current_image_tk, text="")
        except Exception as e:
            self.log(f"[GUI] ERROR loading image: {e}")
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            return

        try:
            predicted_class, confidence, probs = self.client.predict(image_path)
            self.log(f"[GUI] Prediction: {predicted_class} ({confidence:.4f})")

            explanation = self.client.explanation_for_class(predicted_class)
            conf_percent = confidence * 100.0

            self.pred_label_var.set(predicted_class.upper())

            if probs and self.client.labels:
                lines = []
                for label, p in zip(self.client.labels, probs):
                    lines.append(f"{label}: {p * 100:.2f}%")
                probs_str = "\n".join(lines)
            else:
                probs_str = "N/A"

            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert(
                "end",
                f"Image: {image_path}\n\n"
                f"Predicted maturity: {predicted_class}\n"
                f"Top-class probability: {conf_percent:.1f}%\n"
                f"Model version: {self.client.model_version if self.client.model_version > 0 else 'N/A'}\n"
                f"CONF_THRESHOLD: {self.client.conf_threshold:.2f}, "
                f"MARGIN_THRESHOLD: {self.client.margin_threshold:.2f}\n\n"
                f"Explanation:\n{explanation}\n\n"
                f"Per-class probabilities:\n{probs_str}\n",
            )
            self.result_text.configure(state="disabled")
        except Exception as e:
            self.log(f"[GUI] ERROR during prediction: {e}")
            messagebox.showerror("Error", f"Failed to analyze image:\n{e}")


if __name__ == "__main__":
    app = HuskTechAppGUI()
    app.mainloop()
