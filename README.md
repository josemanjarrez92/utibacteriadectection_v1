# Bacterial Detection Research Tool

This repository provides a web-based tool for detecting bacterial colonies in petri dish images using a deep learning model (YOLO, TensorFlow.js). It is designed for research and clinical assessment of urinary tract infection (UTI) bacteria.

## Features
- **Upload Images:** Drag-and-drop or select petri dish images for analysis.
- **Automated Detection:** Uses a pre-trained YOLO model (TensorFlow.js) to detect and classify bacterial colonies.
- **Visual Results:** Overlays detection results on the image, showing bounding boxes and class labels.
- **Clinical Assessment:** Provides infection risk, colony density, and recommendations based on detection results.
- **Detailed Output:** Displays detection summary, confidence scores, and processing time.

## Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Edge, etc.)
- No installation required; runs entirely in the browser.

### Usage
1. **Clone or Download** this repository.
2. **Model Setup:**
   - Place your TensorFlow.js model files (`model.json` and weights) in the `models/` directory.
   - Ensure the model is compatible with the configuration in `app.js` (see `CONFIG.MODEL_URL`).
3. **Open** `index.html` in your browser.
4. **Upload** a petri dish image using the upload section.
5. **View Results:**
   - Detected colonies and their classes will be displayed on the image.
   - Clinical assessment and detailed detection results are shown below.

## File Structure
- `index.html` — Main web interface for uploading images and viewing results.
- `app.js` — JavaScript logic for model loading, image processing, detection, and UI updates.
- `models/` — Directory for TensorFlow.js model files (`model.json`, weights, and `metadata.yaml`).
- `test-images/` — Example images for testing the tool.

## Supported Bacterial Classes
The tool is configured for 10 classes (see `app.js`):
- Candida albicans
- Enterococcus faecalis
- Escherichia coli
- Klebsiella pneumoniae
- Other organisms
- Pseudomonas aeruginosa
- Staphylococcus aureus
- Staphylococcus epidermidis
- Staphylococcus saprophyticus
- Streptococcus agalactiae

## Customization
- **Model:** Replace the model in `models/` with your own trained YOLO model (TensorFlow.js format).
- **Class Names:** Update the `CLASS_NAMES` array in `app.js` if your model uses different classes.
- **Thresholds:** Adjust `CONFIDENCE_THRESHOLD` in `app.js` for your application needs.

## License
See [LICENSE](LICENSE) for details.

## Acknowledgments
- Built with [TensorFlow.js](https://www.tensorflow.org/js) and [Ultralytics YOLO](https://github.com/ultralytics/yolov5).
- Developed for research in UTI bacterial detection and clinical assessment.
