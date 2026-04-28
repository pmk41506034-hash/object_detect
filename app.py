from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    # YOLO() will download weights if missing.
    return YOLO(model_name)


def preprocess_image(pil_image: Image.Image, max_size: int = 960) -> Tuple[np.ndarray, Image.Image]:
    """
    Returns:
      - BGR numpy image for OpenCV drawing (uint8)
      - resized PIL image used for inference (RGB)
    """
    img = pil_image.convert("RGB")
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.BILINEAR)

    rgb = np.array(img)  # (H,W,3) RGB uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, img


def run_detection(
    model: YOLO,
    inference_pil_rgb: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> List[Detection]:
    results = model.predict(
        source=inference_pil_rgb,
        device="cpu",
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False,
    )
    if not results:
        return []

    r0 = results[0]
    names: Dict[int, str] = r0.names if hasattr(r0, "names") else {}

    dets: List[Detection] = []
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls

    xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
    cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)

    for (x1, y1, x2, y2), c, k in zip(xyxy_np, conf_np, cls_np):
        k_int = int(k)
        label = names.get(k_int, str(k_int))
        dets.append(
            Detection(
                label=label,
                confidence=float(c),
                box_xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )
    return dets


def draw_boxes(
    bgr_image: np.ndarray,
    detections: List[Detection],
    box_thickness: int = 2,
) -> np.ndarray:
    out = bgr_image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        color = (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1, th + baseline + 4)
        cv2.rectangle(out, (x1, y_text - th - baseline - 4), (x1 + tw + 8, y_text), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 4, y_text - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def display_results(
    annotated_bgr: np.ndarray,
    detections: List[Detection],
) -> None:
    rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Detection result", use_container_width=True)

    if not detections:
        st.info("No objects detected with the current thresholds.")
        return

    st.subheader("Detected objects")
    counts = Counter([d.label for d in detections])
    total = len(detections)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Total", total)
    with c2:
        st.write(", ".join([f"**{k}**: {v}" for k, v in counts.most_common()]))

    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(sorted(detections, key=lambda x: x.confidence, reverse=True), start=1):
        x1, y1, x2, y2 = d.box_xyxy
        rows.append(
            {
                "#": i,
                "label": d.label,
                "confidence": round(d.confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _read_image_from_streamlit_file(file_obj: Any) -> Optional[Image.Image]:
    if file_obj is None:
        return None
    try:
        return Image.open(file_obj)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Mobile Object Detection (YOLOv8)", layout="wide")

    st.title("Mobile Object Detection App")
    st.write(
        "Capture an image with your phone camera or upload an image, then run **YOLOv8** object detection on **CPU**."
    )

    with st.sidebar:
        st.header("Settings")
        max_size = st.slider("Resize long side (px)", min_value=320, max_value=1280, value=960, step=32)
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU threshold", 0.05, 0.95, 0.45, 0.05)
        max_det = st.slider("Max detections", 10, 500, 300, 10)
        st.caption("Tip: smaller resize improves speed on CPU.")

    st.subheader("Input")
    col_a, col_b = st.columns(2)
    with col_a:
        cam = st.camera_input("Camera capture")
    with col_b:
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

    input_img = _read_image_from_streamlit_file(cam) or _read_image_from_streamlit_file(up)
    if input_img is None:
        st.warning("Provide an image using the camera or upload to start.")
        return

    st.subheader("Preview")
    st.image(input_img, caption="Input image", use_container_width=True)

    run = st.button("Run detection", type="primary", use_container_width=True)
    if not run:
        return

    with st.spinner("Loading model (first run may download weights)..."):
        model = load_model("yolov8n.pt")

    bgr, inference_pil = preprocess_image(input_img, max_size=max_size)

    with st.spinner("Running inference on CPU..."):
        dets = run_detection(model, inference_pil, conf_threshold=conf, iou_threshold=iou, max_det=max_det)

    annotated = draw_boxes(bgr, dets)
    st.subheader("Results")
    display_results(annotated, dets)


if __name__ == "__main__":
    main()

