# app.py
# Otoscopic Classifier — MobileNetV3-Large + Grad-CAM (pytorch-grad-cam only)

import os
import json
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import streamlit as st

# pytorch-grad-cam (NOT torchcam)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -----------------------------
# Config
# -----------------------------
CLASSES_JSON = os.environ.get("CLASSES_JSON", "classes.json")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# -----------------------------
# Default weights resolver
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_ENV = os.environ.get("WEIGHTS_PATH", "").strip()
CANDIDATES = [
    DEFAULT_WEIGHTS_ENV,
    os.path.join(APP_DIR, "weights", "best.pt"),
    os.path.join(APP_DIR, "runs", "classify", "mobilenet_v3_large", "weights", "best.pt"),
    os.path.join(APP_DIR, "..", "weights", "best.pt"),
]
RESOLVED_DEFAULT_WEIGHTS = next((p for p in CANDIDATES if p and os.path.exists(p)), "weights/best.pt")
DEFAULT_WEIGHTS = RESOLVED_DEFAULT_WEIGHTS  # alias used by the sidebar


# -----------------------------
# Helpers
# -----------------------------
def preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_last_conv_layer(module: nn.Module) -> nn.Module:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found.")
    return last


def infer_num_classes_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    # Common MobileNetV3 head keys
    for k in ("classifier.3.weight", "classifier.4.weight", "classifier.2.weight"):
        if k in sd:
            return sd[k].shape[0]
    # Fallback: any small classifier weight in the head
    for k, v in sd.items():
        if k.endswith(".weight") and getattr(v, "ndim", 0) == 2 and "classifier" in k and v.shape[0] <= 100:
            return v.shape[0]
    raise RuntimeError("Could not infer num_classes from checkpoint.")


def build_model(num_classes: int) -> nn.Module:
    # Avoid downloading ImageNet weights; your checkpoint already has everything
    model = models.mobilenet_v3_large(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


def extract_state_dict(state_obj):
    # Handle various save styles
    if isinstance(state_obj, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in state_obj and isinstance(state_obj[key], dict):
                return state_obj[key]
    return state_obj


@st.cache_resource(show_spinner=False)
def load_model_and_classes(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(weights_path, map_location=device)
    state = extract_state_dict(raw)

    num_classes = infer_num_classes_from_state_dict(state)
    model = build_model(num_classes).to(device)

    # Strict load for safety
    model.load_state_dict(state, strict=True)
    model.eval()

    # Class names
    classes = [f"class_{i}" for i in range(num_classes)]
    if CLASSES_JSON and os.path.exists(CLASSES_JSON):
        try:
            with open(CLASSES_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == num_classes:
                classes = [str(x) for x in data]
            elif isinstance(data, dict) and "classes" in data and len(data["classes"]) == num_classes:
                classes = [str(x) for x in data["classes"]]
            else:
                st.warning("classes.json length does not match model head; using generic names.")
        except Exception as e:
            st.warning(f"Failed to read classes.json: {e}")

    return model, classes, device


@st.cache_resource(show_spinner=False)
def get_cam(model: nn.Module):
    return GradCAM(model=model, target_layers=[get_last_conv_layer(model)])


def predict_and_cam(img_pil: Image.Image,
                    model: nn.Module,
                    device: torch.device,
                    classes: List[str],
                    cam_target: str,
                    overlay_alpha: float):
    tfm = preprocess()
    x = tfm(img_pil.convert("RGB")).unsqueeze(0).to(device)

    # Quick prediction
    with torch.no_grad():
        logits = model(x)
    pred_idx = int(logits.argmax(1).item())
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    class_names = classes if len(classes) == probs.shape[0] else [f"class_{i}" for i in range(probs.shape[0])]
    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # Prepare for CAM (needs gradient through the chosen layer)
    x.requires_grad_(True)
    cam = get_cam(model)
    target_idx = pred_idx if cam_target == "pred" or cam_target not in class_names else class_names.index(cam_target)

    img_resized = img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_idx)])[0]
    cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=1.0 - overlay_alpha)
    cam_pil = Image.fromarray(cam_img)

    return class_names[pred_idx], probs_dict, img_resized, cam_pil


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Otoscopic Classifier (MobileNetV3-Large + Grad-CAM)", layout="wide")
st.title("Otoscopic Classifier — MobileNetV3-Large + Grad-CAM")

with st.sidebar:
    st.header("Settings")
    weights_path = st.text_input("Weights path", value=DEFAULT_WEIGHTS)
    show_cam = st.checkbox("Show Grad-CAM overlay", value=True)
    cam_for = st.text_input("Grad-CAM target (class name or 'pred')", value="pred")
    overlay_alpha = st.slider("Overlay strength", 0.0, 1.0, 0.45, 0.05)
    top_k = st.slider("Show top-K probabilities", 1, 10, 5, 1)
    # Optional: keep CPU threads modest on shared hosts
    torch.set_num_threads(2)

# Ensure file exists before load (so you can correct the path in the UI)
if not os.path.exists(weights_path):
    st.error(f"Weights not found at: {weights_path}")
    st.caption(f"Tried defaults: {CANDIDATES}")
    st.stop()

# Load model
try:
    with st.spinner("Loading model..."):
        model, classes, device = load_model_and_classes(weights_path)
    st.success(f"Model loaded on {device}. Classes: {len(classes)}")
except Exception as e:
    st.error(str(e))
    st.stop()

# Uploader
uploads = st.file_uploader("Upload otoscopic image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploads:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Results")
    with c2:
        st.subheader("Grad-CAM" if show_cam else "Preview")

    for up in uploads:
        try:
            img = Image.open(up).convert("RGB")
        except Exception as e:
            st.error(f"Failed to open {up.name}: {e}")
            continue

        pred, probs_dict, preview_img, cam_img = predict_and_cam(
            img, model, device, classes, cam_target=(cam_for.strip() or "pred"), overlay_alpha=overlay_alpha
        )

        # Top-K table
        items = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)[:min(top_k, len(probs_dict))]
        table = {"class": [k for k, _ in items], "probability": [float(v) for _, v in items]}

        a, b = st.columns(2)
        with a:
            st.markdown(f"**{up.name}**")
            st.image(preview_img, use_column_width=True)
            st.write(f"**Prediction:** {pred}")
            st.table(table)
        with b:
            if show_cam:
                st.image(cam_img, caption=f"Grad-CAM ({cam_for})", use_column_width=True)
            else:
                st.image(preview_img, use_column_width=True)

