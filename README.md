## Mobile Object Detection App (Streamlit + YOLOv8)

Streamlit web app that detects objects from:
- Smartphone camera capture (`st.camera_input`)
- Image upload (`st.file_uploader`)

Outputs:
- Bounding boxes on the image
- Label + confidence
- Object count statistics

### Setup

From the project folder:

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

### Notes

- **Model**: pretrained Ultralytics YOLOv8 nano (`yolov8n.pt`)
- **CPU-only**: inference is forced to CPU
- **Performance**: images are resized before inference (configurable in the UI)
