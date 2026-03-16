# Bacterial Detection Research Tool

This repository provides a web-based tool for detecting bacterial colonies in petri dish images using deep learning models (YOLO, TensorFlow.js). It is designed for research and clinical assessment of urinary tract infection (UTI) bacteria, with support for multiple model variants for performance comparison.

## Features

- **Multi-Model Support:** Switch between 4 YOLO variants (YOLO26n, YOLO26s, YOLO26m, YOLO26l) to compare speed vs. accuracy.
- **Upload Images:** Drag-and-drop or select petri dish images for analysis.
- **Automated Detection:** Uses pre-trained YOLO models (TensorFlow.js) to detect and classify bacterial colonies.
- **Automatic Reprocessing:** Process the same image with different models instantly without re-uploading.
- **Visual Results:** Overlays detection results on the image with bounding boxes and interactive class labels (hover for details).
- **Performance Metrics:** Tracks and displays model loading time and image processing time in the console for performance comparison.
- **Clinical Assessment:** Provides infection risk, colony density, and recommendations based on detection results.
- **Detailed Output:** Displays detection summary, confidence scores, average confidence, and processing time.

## Getting Started

### Prerequisites

- Modern web browser (Chrome, Firefox, Edge, Safari, etc.)
- No installation required; runs entirely in the browser.

### Usage

1. **Clone or Download** this repository.
2. **Model Setup:**
   - Place your TensorFlow.js model files in the `models/` directory with the following structure:
     ```
     models/
     ├── yolo26n/
     │   ├── model.json
     │   └── weights (binary files)
     ├── yolo26s/
     │   ├── model.json
     │   └── weights
     ├── yolo26m/
     │   ├── model.json
     │   └── weights
     └── yolo26l/
         ├── model.json
         └── weights
     ```
   - Update model paths in `app.js` if your directory structure differs (see `MODELS` configuration).
   - Ensure models are compatible with the configuration in `app.js`.

3. **Open** `Bacterial_Detection_Browser_App.html` in your browser (or rename to `index.html` if preferred).

4. **Select Model:** Choose your preferred model from the dropdown (YOLO26n recommended for speed, YOLO26l for accuracy).

5. **Upload** a petri dish image using the upload section.

6. **View Results:**
   - Detected colonies and their classes will be displayed on the image.
   - Clinical assessment and detailed detection results are shown below.

7. **Compare Models:** Switch to a different model in the dropdown to automatically reprocess the same image with the new model—no re-upload needed.

## File Structure

- `Bacterial_Detection_Browser_App.html` — Main web interface for uploading images and viewing results.
- `app.js` — JavaScript logic for model loading, image processing, detection, UI updates, and automatic reprocessing.
- `models/` — Directory for TensorFlow.js model files (YOLO26n, YOLO26s, YOLO26m, YOLO26l).
- `test-images/` — Example images for testing the tool.

## Model Selection

The tool supports 4 YOLO variants with different speed/accuracy trade-offs:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **YOLO26n** | Fastest (~400ms) | Good | Real-time analysis |
| **YOLO26s** | Balanced (~550ms) | Better | General use |
| **YOLO26m** | Slower (~850ms) | Very Good | Research analysis |
| **YOLO26l** | Slowest (~1250ms) | Best | High-accuracy research |

*Processing times are approximate and depend on image size and hardware.*

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

## Performance Monitoring

The tool includes built-in performance tracking:

- **Model Loading Time:** Time to load the selected model (visible in browser console)
- **Image Processing Time:** Time to process the image with the current model (displayed in results and console)
- **Console Logging:** Detailed logs show when models are loaded and images are processed, including timestamps

To view performance metrics:
1. Open browser Developer Tools (F12)
2. Go to the Console tab
3. Upload an image or switch models to see timing information

Example console output:
```
⏱️ MODEL LOADING TIME: 1250.45ms (1.250s)
⏱️ IMAGE PROCESSING TIME: 425.67ms (0.425s)
```

## Automatic Image Reprocessing

The tool uses a reference-based approach for efficient image handling:

- **Upload Once:** Upload a petri dish image once, then switch between models without re-uploading.
- **Memory Efficient:** Stores a reference to the image (not a copy), saving memory and enabling rapid model switching.
- **Automatic Processing:** When you change the model, the same image is automatically processed with the new model.

This is particularly useful for research comparing model performance on identical samples.

## Customization

- **Model Paths:** Update the `MODELS` object in `app.js` if your models are in different locations:
  ```javascript
  const MODELS = {
      yolo26n: { path: './models/yolo26n/model.json', name: 'YOLO26n (Fast)' },
      // ... etc
  };
  ```

- **Class Names:** Update the `CLASS_NAMES` array in `app.js` if your models use different classes.

- **Confidence Threshold:** Adjust `CONFIDENCE_THRESHOLD` in `app.js` for your application needs (default: 0.25 for medical applications).

- **Input Size:** Modify `INPUT_SIZE` in `app.js` if your models expect different input dimensions (default: 640x640).

## Research Applications

The multi-model support and automatic reprocessing enable several research workflows:

1. **Model Comparison:** Upload a single sample and test all models to compare detection sensitivity
2. **Speed vs. Accuracy Analysis:** Compare processing time and detection accuracy across models
3. **Batch Testing:** Test multiple samples with all models for comprehensive performance evaluation
4. **Model Selection:** Make data-driven decisions about which model best suits your research needs

## Browser Console Commands (Developer Mode)

For advanced users, debug functions are available in the browser console:

```javascript
// View the current model
debugFunctions.getCurrentModel()

// View all available models
debugFunctions.getAvailableModels()

// Check memory usage
debugFunctions.logMemory()

// Download results as JSON
debugFunctions.downloadResults(detections)

// Test detection pipeline
await debugFunctions.testDetectionPipeline()
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [TensorFlow.js](https://www.tensorflow.org/js) and [Ultralytics YOLO](https://github.com/ultralytics/yolov5).
- Developed for research in UTI bacterial detection and clinical assessment.
- Reference-based image processing approach enables efficient multi-model comparison.
