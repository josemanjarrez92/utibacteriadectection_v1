// Bacterial Detection Research Tool - JavaScript Implementation
// Global variables
let model = null;
let isModelLoaded = false;

// Configuration for bacterial detection model
const CONFIG = {
    MODEL_URL: './models/model.json', // Update this path to your model
    CONFIDENCE_THRESHOLD: 0.25,       // 25% confidence threshold for medical applications
    INPUT_SIZE: 640,                  // YOLO standard input size (adjust based on your export)
    
    // Your 10 bacterial classes (adjust order if needed)
    CLASS_NAMES: [
        'Candida_albicans',           // Class 0
        'Enterococcus_faecalis',      // Class 1
        'Escherichia_coli',           // Class 2
        'Klebsiella_pneumoniae',      // Class 3
        'Other_organisms',            // Class 4
        'Pseudomonas_aeruginosa',     // Class 5
        'Staphylococcus_aureus',      // Class 6
        'Staphylococcus_epidermidis', // Class 7
        'Staphylococcus_saprophyticus', // Class 8
        'Streptococcus_agalactiae'    // Class 9
    ],
    
    COLORS: [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
        '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A'
    ], // One color per bacterial class
    
    // Debugging options
    DEBUG: {
        VERBOSE_LOGGING: true,
        LOG_RAW_OUTPUTS: false,
        LOG_ALL_DETECTIONS: false,
        SHOW_COORDINATE_ANALYSIS: true,
        SHOW_SCALING_INFO: true  // Show coordinate scaling details
    }
};

// Helper function to auto-detect and fix class mapping issues
function detectClassMappingIssues(modelNumClasses, detectedClasses) {
    console.log('🔍 Analyzing class mapping...');
    
    const configNumClasses = CONFIG.CLASS_NAMES.length;
    
    if (modelNumClasses !== configNumClasses) {
        console.warn(`⚠️ Class count mismatch detected!`);
        console.warn(`Model expects: ${modelNumClasses} classes`);
        console.warn(`Config has: ${configNumClasses} classes`);
        
        // Common scenarios
        if (modelNumClasses === configNumClasses + 1) {
            console.log('💡 Model might include a background class');
            console.log('Try updating CONFIG.CLASS_NAMES to include "background" as class 0');
        } else if (modelNumClasses === 80) {
            console.warn('❌ Model still uses COCO classes (80). Need to re-export with correct classes.');
        } else if (modelNumClasses > configNumClasses) {
            console.log(`💡 Model has ${modelNumClasses - configNumClasses} extra classes`);
            console.log('Add missing class names to CONFIG.CLASS_NAMES or verify export');
        }
        
        return false;
    }
    
    // Check if detected classes match expected range
    const maxDetectedClass = Math.max(...detectedClasses);
    if (maxDetectedClass >= configNumClasses) {
        console.warn(`⚠️ Detected class ID ${maxDetectedClass} but only have ${configNumClasses} class names`);
        return false;
    }
    
    console.log('✅ Class mapping looks correct');
    return true;
}

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🔬 Bacterial Detection Tool Starting...');
    console.log('TensorFlow.js version:', tf.version.tfjs);
    
    setupEventListeners();
    await loadModel();
});

// Set up event listeners
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadSection = document.getElementById('uploadSection');
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadSection.addEventListener('dragover', handleDragOver);
    uploadSection.addEventListener('dragleave', handleDragLeave);
    uploadSection.addEventListener('drop', handleDrop);
}

// Load the TensorFlow.js model (GraphDef format from Ultralytics)
async function loadModel() {
    updateModelStatus('Loading model...', 'info');
    
    try {
        console.log('📥 Loading GraphModel from:', CONFIG.MODEL_URL);
        
        // Load the GraphModel (for Ultralytics exported models)
        model = await tf.loadGraphModel(CONFIG.MODEL_URL);
        
        console.log('✅ Model loaded successfully');
        
        // For GraphModels, we need to inspect the model differently
        console.log('Model signature:', Object.keys(model.modelSignature || {}));
        console.log('Model inputs:', model.inputs?.map(input => input.shape) || 'Not available');
        console.log('Model outputs:', model.outputs?.map(output => output.shape) || 'Not available');
        
        // Warm up the model with a dummy prediction
        await warmUpModel();
        
        isModelLoaded = true;
        updateModelStatus('Model loaded and ready', 'success');
        
    } catch (error) {
        console.error('❌ Error loading model:', error);
        updateModelStatus(`Failed to load model: ${error.message}`, 'error');
    }
}

// Warm up the model with a dummy prediction (TensorFlow.js format with async execution)
async function warmUpModel() {
    console.log('🔥 Warming up model...');
    
    // TensorFlow.js YOLO models expect [batch, height, width, channels] format
    const inputSize = CONFIG.INPUT_SIZE;
    const dummyInput = tf.randomNormal([1, inputSize, inputSize, 3]);
    
    try {
        console.log('Warmup input shape:', dummyInput.shape);
        // Use executeAsync for models with NMS
        const warmupPrediction = await model.executeAsync(dummyInput);
        
        // Handle multiple outputs (YOLO typically has 1 or 3 outputs)
        if (Array.isArray(warmupPrediction)) {
            console.log('Model outputs count:', warmupPrediction.length);
            warmupPrediction.forEach((output, i) => {
                console.log(`Output ${i} shape:`, output.shape);
                output.dispose();
            });
        } else {
            console.log('Model output shape:', warmupPrediction.shape);
            warmupPrediction.dispose();
        }
        
        dummyInput.dispose();
        console.log('✅ Model warmed up successfully');
    } catch (error) {
        console.error('❌ Model warmup failed:', error);
        dummyInput.dispose();
        throw error;
    }
}

// Update model status display
function updateModelStatus(message, type) {
    const statusEl = document.getElementById('modelStatus');
    const statusTextEl = document.getElementById('modelStatusText');
    
    statusTextEl.textContent = message;
    statusEl.className = `status ${type}`;
    statusEl.style.display = 'block';
    
    // Hide status after 5 seconds if it's a success message
    if (type === 'success') {
        setTimeout(() => {
            statusEl.style.display = 'none';
        }, 5000);
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImageFile(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processImageFile(files[0]);
    }
}

// Process the uploaded image file
async function processImageFile(file) {
    if (!isModelLoaded) {
        alert('Please wait for the model to load before processing images.');
        return;
    }
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }
    
    console.log('📸 Processing image:', file.name);
    showLoading(true);
    
    try {
        // Create image element
        const img = await createImageElement(file);
        
        // Display the image
        displayImage(img);
        
        // Run detection
        const startTime = performance.now();
        const detections = await runDetection(img);
        const endTime = performance.now();
        const processingTime = endTime - startTime;
        
        // Display results
        displayResults(detections, processingTime);
        
        // Draw detection boxes
        drawDetections(img, detections);
        
        console.log('✅ Detection completed successfully');
        
    } catch (error) {
        console.error('❌ Error processing image:', error);
        alert(`Error processing image: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Create image element from file
function createImageElement(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

// Display the uploaded image
function displayImage(img) {
    const preview = document.getElementById('imagePreview');
    const container = document.getElementById('canvasContainer');
    
    // Clean up any existing tooltip
    const existingTooltip = document.getElementById('detection-tooltip');
    if (existingTooltip) {
        existingTooltip.style.display = 'none';
    }
    
    preview.src = img.src;
    preview.style.display = 'block';
    container.style.display = 'block';
    
    // Setup canvas for drawing detections
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match image
    canvas.width = preview.naturalWidth || img.width;
    canvas.height = preview.naturalHeight || img.height;
    
    // Scale canvas to match displayed image size
    const rect = preview.getBoundingClientRect();
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
}

// Preprocess image for TensorFlow.js YOLO model input
function preprocessImage(img) {
    return tf.tidy(() => {
        console.log('🔄 Preprocessing image for TensorFlow.js YOLO model...');
        
        // Convert image to tensor (browser gives us [H, W, C] in RGB format)
        let tensor = tf.browser.fromPixels(img, 3); // Ensure 3 channels (RGB)
        console.log('Original tensor shape:', tensor.shape);
        
        // Resize to model's expected input size
        const inputSize = CONFIG.INPUT_SIZE;
        const resized = tf.image.resizeBilinear(tensor, [inputSize, inputSize]);
        console.log('Resized tensor shape:', resized.shape);
        
        // Normalize to [0, 1] range
        const normalized = resized.div(255.0);
        
        // TensorFlow.js YOLO models expect NHWC format [batch, height, width, channels]
        // So we keep the [H, W, C] format and just add batch dimension
        const batched = normalized.expandDims(0);
        
        console.log('✅ Preprocessing complete. Final shape:', batched.shape);
        console.log('Expected model input shape: [1, 640, 640, 3]');
        return batched;
    });
}

// Run detection on the preprocessed image
async function runDetection(img) {
    console.log('🔍 Running bacterial detection...');
    
    // Store original image dimensions for coordinate scaling
    const originalWidth = img.width;
    const originalHeight = img.height;
    
    console.log(`Original image size: ${originalWidth}x${originalHeight}`);
    console.log(`Model input size: ${CONFIG.INPUT_SIZE}x${CONFIG.INPUT_SIZE}`);
    
    // Preprocess the image
    const inputTensor = preprocessImage(img);
    
    try {
        // Run inference using executeAsync for models with NMS
        console.log('🧠 Running model inference (async)...');
        const predictions = await model.executeAsync(inputTensor);
        
        // Process the predictions with original dimensions for proper scaling
        const detections = await processDetections(predictions, {
            width: originalWidth,
            height: originalHeight,
            modelInputSize: CONFIG.INPUT_SIZE
        });
        
        // Clean up tensors
        inputTensor.dispose();
        if (Array.isArray(predictions)) {
            predictions.forEach(pred => pred.dispose());
        } else {
            predictions.dispose();
        }
        
        console.log(`✅ Detection complete. Found ${detections.length} bacterial colonies`);
        return detections;
        
    } catch (error) {
        inputTensor.dispose();
        throw error;
    }
}

// Enhanced detection processing for NMS-enabled models
async function processDetections(predictions, imageInfo) {
    console.log('📊 Processing TensorFlow.js YOLO detection results (NMS-enabled)...');
    
    const originalWidth = imageInfo.width;
    const originalHeight = imageInfo.height;
    const modelInputSize = imageInfo.modelInputSize;
    
    console.log(`Image scaling: ${originalWidth}x${originalHeight} → ${modelInputSize}x${modelInputSize} → back to ${originalWidth}x${originalHeight}`);
    
    // Handle multiple outputs
    let outputTensor;
    if (Array.isArray(predictions)) {
        console.log('Multiple outputs detected:', predictions.length);
        predictions.forEach((pred, i) => {
            console.log(`Output ${i} shape:`, pred.shape);
        });
        outputTensor = predictions[0];
        // Note: Don't dispose here as they're disposed in the calling function
    } else {
        outputTensor = predictions;
    }
    
    // Run comprehensive analysis first
    await comprehensiveOutputAnalysis(predictions);
    
    const predictionsData = await outputTensor.data();
    const shape = outputTensor.shape;
    const detections = [];
    
    const batchSize = shape[0];
    const numDetections = shape[1];
    const outputSize = shape[2];
    
    console.log(`\n📋 Processing NMS detections:`);
    console.log(`Batch size: ${batchSize}`);
    console.log(`Number of possible detections: ${numDetections}`);
    console.log(`Output size per detection: ${outputSize}`);
    
    // Determine the output format based on size
    let formatType = 'unknown';
    
    if (outputSize === 6) {
        formatType = 'nms_6'; // [x1, y1, x2, y2, confidence, class_id]
        console.log('🔍 Detected format: NMS 6-value [x1, y1, x2, y2, confidence, class_id]');
    } else if (outputSize === 7) {
        formatType = 'nms_7'; // [x1, y1, x2, y2, confidence, class_id, valid] or similar
        console.log('🔍 Detected format: NMS 7-value format');
    } else if (outputSize >= 15) {
        formatType = 'raw_yolo'; // Still raw format despite NMS
        console.log('🔍 Detected format: Raw YOLO format (NMS might not be applied)');
    } else {
        console.log('🔍 Detected format: Unknown - will attempt best guess');
    }
    
    // Process each detection based on detected format
    let validDetections = 0;
    const detectedClasses = [];
    
    for (let i = 0; i < numDetections; i++) {
        const startIdx = i * outputSize;
        
        // Extract detection data based on format
        let x1, y1, x2, y2, classId, confidence;
        let isValid = false;
        let topClasses = [];
        if (formatType === 'nms_6') {
            // Format: [x1, y1, x2, y2, confidence, class_id] (corrected!)
            x1 = predictionsData[startIdx];
            y1 = predictionsData[startIdx + 1];
            x2 = predictionsData[startIdx + 2];
            y2 = predictionsData[startIdx + 3];
            confidence = predictionsData[startIdx + 4];  // Confidence is at position 4
            classId = Math.round(predictionsData[startIdx + 5]);  // Class ID is at position 5
            isValid = confidence > 0.001;
            // Only top-1 available in this format
            topClasses = [{
                class_id: classId,
                class_name: CONFIG.CLASS_NAMES[classId] || `Class_${classId}`,
                confidence: confidence
            }];
        } else if (formatType === 'nms_7') {
            // Try common format: [x1, y1, x2, y2, confidence, class_id, valid]
            x1 = predictionsData[startIdx];
            y1 = predictionsData[startIdx + 1];
            x2 = predictionsData[startIdx + 2];
            y2 = predictionsData[startIdx + 3];
            confidence = predictionsData[startIdx + 4];
            classId = Math.round(predictionsData[startIdx + 5]);
            const validFlag = predictionsData[startIdx + 6];
            isValid = confidence > 0.001 && (validFlag === undefined || validFlag > 0);
            topClasses = [{
                class_id: classId,
                class_name: CONFIG.CLASS_NAMES[classId] || `Class_${classId}`,
                confidence: confidence
            }];
        } else {
            // Fall back to raw YOLO processing
            const x_center = predictionsData[startIdx];
            const y_center = predictionsData[startIdx + 1]; 
            const width = predictionsData[startIdx + 2];
            const height = predictionsData[startIdx + 3];
            const objectness_logit = predictionsData[startIdx + 4];
            const objectness = 1 / (1 + Math.exp(-objectness_logit)); // Sigmoid
            if (objectness < 0.01) continue;
            const numClasses = outputSize - 5;
            // Compute all class probabilities
            let classProbs = [];
            for (let c = 0; c < Math.min(numClasses, CONFIG.CLASS_NAMES.length); c++) {
                const classLogit = predictionsData[startIdx + 5 + c];
                const classProb = 1 / (1 + Math.exp(-classLogit)); // Sigmoid
                classProbs.push({
                    class_id: c,
                    class_name: CONFIG.CLASS_NAMES[c] || `Class_${c}`,
                    confidence: objectness * classProb
                });
            }
            // Sort and take top 3
            classProbs.sort((a, b) => b.confidence - a.confidence);
            topClasses = classProbs.slice(0, 3);
            // Use top-1 for main detection
            classId = topClasses[0].class_id;
            confidence = topClasses[0].confidence;
            // Convert center format to corner format (in model input space)
            x1 = (x_center - width / 2) * modelInputSize;
            y1 = (y_center - height / 2) * modelInputSize;
            x2 = (x_center + width / 2) * modelInputSize;
            y2 = (y_center + height / 2) * modelInputSize;
            isValid = confidence > CONFIG.CONFIDENCE_THRESHOLD;
        }
        
        // Detailed logging for first few detections
        if (i < 5) {
            console.log(`\nDetection ${i + 1} analysis:`);
            console.log(`  Raw values: [${Array.from(predictionsData.slice(startIdx, startIdx + Math.min(8, outputSize))).map(x => x.toFixed(3)).join(', ')}]`);
            console.log(`  Parsed (model space): x1=${x1?.toFixed(1)}, y1=${y1?.toFixed(1)}, x2=${x2?.toFixed(1)}, y2=${y2?.toFixed(1)}`);
            console.log(`  Confidence: ${(confidence * 100).toFixed(1)}% (raw: ${confidence?.toFixed(3)})`);
            console.log(`  Class: ${classId} (${CONFIG.CLASS_NAMES[classId] || 'Unknown'})`);
            console.log(`  Valid: ${isValid}`);
            console.log(`  Above threshold (${(CONFIG.CONFIDENCE_THRESHOLD * 100).toFixed(1)}%): ${confidence > CONFIG.CONFIDENCE_THRESHOLD}`);
            
            // Validate class ID is reasonable
            if (classId < 0 || classId >= CONFIG.CLASS_NAMES.length) {
                console.warn(`  ⚠️ Class ID ${classId} is out of range [0, ${CONFIG.CLASS_NAMES.length-1}]`);
            }
        }
        
        // Apply confidence threshold and validate
        if (isValid && confidence > CONFIG.CONFIDENCE_THRESHOLD) {
            // Validate coordinates
            const coordsValid = !isNaN(x1) && !isNaN(y1) && !isNaN(x2) && !isNaN(y2);
            if (!coordsValid) {
                console.warn(`⚠️ Invalid coordinates in detection ${i + 1}`);
                continue;
            }
            
            // Handle coordinate scaling properly
            let scaledX1, scaledY1, scaledX2, scaledY2;
            
            if (formatType.startsWith('nms')) {
                // For NMS models, coordinates might be in different formats
                if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.1) {
                    // Coordinates are normalized [0,1], scale to original image size
                    scaledX1 = x1 * originalWidth;
                    scaledY1 = y1 * originalHeight;
                    scaledX2 = x2 * originalWidth;
                    scaledY2 = y2 * originalHeight;
                    if (CONFIG.DEBUG.SHOW_SCALING_INFO && i < 3) {
                        console.log(`  Scaling normalized coords to original: [${x1.toFixed(3)}, ${y1.toFixed(3)}, ${x2.toFixed(3)}, ${y2.toFixed(3)}] → [${scaledX1.toFixed(1)}, ${scaledY1.toFixed(1)}, ${scaledX2.toFixed(1)}, ${scaledY2.toFixed(1)}]`);
                    }
                } else if (Math.max(x1, y1, x2, y2) <= modelInputSize * 1.1) {
                    // Coordinates are in model input space (e.g., 640x640), scale to original image size
                    scaledX1 = (x1 / modelInputSize) * originalWidth;
                    scaledY1 = (y1 / modelInputSize) * originalHeight;
                    scaledX2 = (x2 / modelInputSize) * originalWidth;
                    scaledY2 = (y2 / modelInputSize) * originalHeight;
                    if (CONFIG.DEBUG.SHOW_SCALING_INFO && i < 3) {
                        console.log(`  Scaling model coords to original: [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}] → [${scaledX1.toFixed(1)}, ${scaledY1.toFixed(1)}, ${scaledX2.toFixed(1)}, ${scaledY2.toFixed(1)}]`);
                    }
                } else {
                    // Coordinates might already be in original image space
                    scaledX1 = x1;
                    scaledY1 = y1;
                    scaledX2 = x2;
                    scaledY2 = y2;
                    if (CONFIG.DEBUG.SHOW_SCALING_INFO && i < 3) {
                        console.log(`  Using coordinates as-is (already in original space)`);
                    }
                }
            } else {
                // For raw YOLO, coordinates are already scaled from center format
                scaledX1 = (x1 / modelInputSize) * originalWidth;
                scaledY1 = (y1 / modelInputSize) * originalHeight;
                scaledX2 = (x2 / modelInputSize) * originalWidth;
                scaledY2 = (y2 / modelInputSize) * originalHeight;
            }
            
            // Ensure valid bounding box within image bounds
            scaledX1 = Math.max(0, Math.min(scaledX1, scaledX2));
            scaledY1 = Math.max(0, Math.min(scaledY1, scaledY2));
            scaledX2 = Math.min(originalWidth, Math.max(scaledX1, scaledX2));
            scaledY2 = Math.min(originalHeight, Math.max(scaledY1, scaledY2));
            
            const width = scaledX2 - scaledX1;
            const height = scaledY2 - scaledY1;
            
            if (width > 5 && height > 5) { // Minimum size check
                const detection = {
                    bbox: { 
                        x1: scaledX1, 
                        y1: scaledY1, 
                        x2: scaledX2, 
                        y2: scaledY2 
                    },
                    class_id: classId,
                    confidence: confidence,
                    class_name: CONFIG.CLASS_NAMES[classId] || `Class_${classId}`,
                    width: width,
                    height: height,
                    area: width * height,
                    center: {
                        x: (scaledX1 + scaledX2) / 2,
                        y: (scaledY1 + scaledY2) / 2
                    },
                    // New: top-3 class predictions
                    top_classes: topClasses,
                    // Debug info
                    format_type: formatType,
                    detection_index: i,
                    original_coords: { x1, y1, x2, y2 },
                    scaled_coords: { x1: scaledX1, y1: scaledY1, x2: scaledX2, y2: scaledY2 }
                };
                detections.push(detection);
                detectedClasses.push(classId);
                validDetections++;
            } else {
                console.warn(`⚠️ Bounding box too small after scaling: ${width.toFixed(1)}x${height.toFixed(1)}`);
            }
        }
    }
    
    // Sort by confidence
    detections.sort((a, b) => b.confidence - a.confidence);
    
    console.log(`\n✅ Final results: ${detections.length} valid detections`);
    if (detections.length > 0) {
        const avgConfidence = detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length;
        console.log(`Average confidence: ${(avgConfidence * 100).toFixed(1)}%`);
        
        // Show class distribution
        const classCounts = {};
        detections.forEach(d => {
            classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
        });
        console.log('Class distribution:', classCounts);
        
        // Check for class mapping issues
        const uniqueClasses = [...new Set(detectedClasses)];
        const maxClass = Math.max(...detectedClasses);
        if (maxClass >= CONFIG.CLASS_NAMES.length) {
            console.error(`❌ Class ${maxClass} detected but only ${CONFIG.CLASS_NAMES.length} class names configured`);
        }
    }
    
    // Special warning for exactly 4 detections
    if (detections.length === 4) {
        console.warn('⚠️ Exactly 4 detections found');
        console.warn('If this always happens, check if model has max_detections=4 setting');
    }
    
    return detections;
}

// Draw detection boxes on the canvas with hover functionality
function drawDetections(img, detections) {
    console.log('🎨 Drawing detection boxes...');
    
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset debug counter for hover detection
    canvas._debugCount = 0;
    
    // Calculate scale factors for DRAWING (canvas internal dimensions)
    const drawScaleX = canvas.width / img.width;
    const drawScaleY = canvas.height / img.height;
    
    // Calculate scale factors for HOVER DETECTION (canvas display dimensions)
    const hoverScaleX = canvas.offsetWidth / img.width;
    const hoverScaleY = canvas.offsetHeight / img.height;
    
    console.log(`Drawing scale factors: ${drawScaleX.toFixed(3)}x${drawScaleY.toFixed(3)}`);
    console.log(`Hover scale factors: ${hoverScaleX.toFixed(3)}x${hoverScaleY.toFixed(3)}`);
    console.log(`Canvas internal size: ${canvas.width}x${canvas.height}`);
    console.log(`Canvas display size: ${canvas.offsetWidth}x${canvas.offsetHeight}`);
    console.log(`Original image size: ${img.width}x${img.height}`);
    
    // Store detection info for hover functionality with DISPLAY coordinates
    canvas.detectionData = detections.map((detection, index) => {
        const displayBox = {
            x1: detection.bbox.x1 * hoverScaleX,
            y1: detection.bbox.y1 * hoverScaleY,
            x2: detection.bbox.x2 * hoverScaleX,
            y2: detection.bbox.y2 * hoverScaleY
        };
        return {
            ...detection,
            displayBox: displayBox
        };
    });
    
    detections.forEach((detection, index) => {
        // Use consistent color per class instead of per detection
        const color = CONFIG.COLORS[detection.class_id % CONFIG.COLORS.length];
        
        // Scale bounding box coordinates for DRAWING (using canvas internal dimensions)
        const x1 = detection.bbox.x1 * drawScaleX;
        const y1 = detection.bbox.y1 * drawScaleY;
        const width = (detection.bbox.x2 - detection.bbox.x1) * drawScaleX;
        const height = (detection.bbox.y2 - detection.bbox.y1) * drawScaleY;
        
        // Draw bounding box only (no text label)
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw center point
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(detection.center.x * drawScaleX, detection.center.y * drawScaleY, 4, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // Add hover functionality
    setupHoverDetection(canvas);
    
    console.log('✅ Detection boxes drawn with hover functionality');
}

// Setup hover detection for showing class names
function setupHoverDetection(canvas) {
    let tooltip = document.getElementById('detection-tooltip');
    
    // Create tooltip if it doesn't exist
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'detection-tooltip';
        tooltip.style.cssText = `
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 14px;
            pointer-events: none;
            z-index: 10000;
            display: none;
            font-family: Arial, sans-serif;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(4px);
        `;
        document.body.appendChild(tooltip);
    }
    
    // Ensure canvas can receive mouse events
    canvas.style.pointerEvents = 'auto';
    canvas.style.cursor = 'default';
    
    // Remove existing listeners
    canvas.removeEventListener('mousemove', canvas._hoverHandler);
    canvas.removeEventListener('mouseleave', canvas._leaveHandler);
    
    // Create new event handlers
    canvas._hoverHandler = function(event) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Debug logging (only for first few mouse moves)
        if (!canvas._debugCount) canvas._debugCount = 0;
        if (canvas._debugCount < 3) {
            console.log(`Mouse position: ${x.toFixed(1)}, ${y.toFixed(1)}`);
            console.log(`Available detections: ${canvas.detectionData?.length || 0}`);
            canvas._debugCount++;
        }
        
        // Check if mouse is over any detection
        const hoveredDetection = canvas.detectionData?.find(det => {
            const box = det.displayBox;
            const isInside = x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2;
            
            // Debug logging for first detection
            if (canvas._debugCount <= 3 && det === canvas.detectionData[0]) {
                console.log(`Checking detection: box=[${box.x1.toFixed(1)}, ${box.y1.toFixed(1)}, ${box.x2.toFixed(1)}, ${box.y2.toFixed(1)}], inside=${isInside}`);
            }
            
            return isInside;
        });
        
        if (hoveredDetection) {
            // Show tooltip with top-3 predictions
            const top3 = (hoveredDetection.top_classes || [{class_name: hoveredDetection.class_name, confidence: hoveredDetection.confidence}])
                .map((c, i) => `<div style=\"color:${CONFIG.COLORS[c.class_id % CONFIG.COLORS.length]}; font-weight:${i === 0 ? 'bold' : 'normal'}\">${c.class_name} (${(c.confidence * 100).toFixed(1)}%)</div>`)
                .join('');
            tooltip.innerHTML = `
                <div style="font-weight: bold; color: ${CONFIG.COLORS[hoveredDetection.class_id % CONFIG.COLORS.length]};">
                    Colony
                </div>
                <div style="font-size: 12px; margin-top: 4px;">
                    ${top3}
                </div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 15) + 'px';
            tooltip.style.top = (event.clientY - 10) + 'px';
            canvas.style.cursor = 'pointer';
        } else {
            tooltip.style.display = 'none';
            canvas.style.cursor = 'default';
        }
    };
    
    canvas._leaveHandler = function() {
        tooltip.style.display = 'none';
        canvas.style.cursor = 'default';
    };
    
    // Add new event listeners
    canvas.addEventListener('mousemove', canvas._hoverHandler);
    canvas.addEventListener('mouseleave', canvas._leaveHandler);
    
    console.log('✅ Hover detection setup complete');
}

// Display detection results
function displayResults(detections, processingTime) {
    console.log('📋 Displaying results...');
    
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Update summary statistics
    document.getElementById('coloniesCount').textContent = detections.length;
    document.getElementById('processingTime').textContent = `${processingTime.toFixed(2)}ms`;
    
    if (detections.length > 0) {
        const avgConfidence = detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length;
        document.getElementById('avgConfidence').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        
        // Clinical assessment
        updateClinicalAssessment(detections);
        
        // Detailed results
        displayDetailedResults(detections);
    } else {
        document.getElementById('avgConfidence').textContent = 'N/A';
        document.getElementById('infectionRisk').textContent = 'Low';
        document.getElementById('colonyDensity').textContent = 'None detected';
        document.getElementById('recommendation').textContent = 'No bacterial colonies detected';
        document.getElementById('detectionList').innerHTML = '<p>No bacterial colonies detected in this image.</p>';
    }
}

// Update clinical assessment based on detections
function updateClinicalAssessment(detections) {
    const coloniesCount = detections.length;
    const avgConfidence = detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length;
    
    // Calculate density (colonies per unit area - this is simplified)
    const totalArea = detections.reduce((sum, d) => sum + d.area, 0);
    const avgColonySize = totalArea / coloniesCount;
    
    // Risk assessment (simplified - adjust based on your clinical criteria)
    let riskLevel, densityLevel, recommendation;
    
    if (coloniesCount === 0) {
        riskLevel = 'Low';
        densityLevel = 'None';
        recommendation = 'No bacterial growth detected';
    } else if (coloniesCount <= 5) {
        riskLevel = 'Low to Moderate';
        densityLevel = 'Light';
        recommendation = 'Light bacterial growth - monitor closely';
    } else if (coloniesCount <= 15) {
        riskLevel = 'Moderate';
        densityLevel = 'Moderate';
        recommendation = 'Moderate bacterial growth - consider treatment';
    } else {
        riskLevel = 'High';
        densityLevel = 'Heavy';
        recommendation = 'Heavy bacterial growth - immediate treatment recommended';
    }
    
    document.getElementById('infectionRisk').textContent = riskLevel;
    document.getElementById('colonyDensity').textContent = densityLevel;
    document.getElementById('recommendation').textContent = recommendation;
}

// Display detailed detection results
function displayDetailedResults(detections) {
    const detailsList = document.getElementById('detectionList');
    let html = '<table style="width: 100%; border-collapse: collapse;">';
    html += '<tr style="background: #f8f9fa;"><th style="padding: 8px; border: 1px solid #ddd;">Colony #</th><th style="padding: 8px; border: 1px solid #ddd;">Top 3 Species (Confidence)</th></tr>';
    detections.forEach((detection, index) => {
        const color = CONFIG.COLORS[detection.class_id % CONFIG.COLORS.length];
        // Always show 3 rows for top-3, even if only 1 or 2 present
        let top3 = detection.top_classes || [{class_name: detection.class_name, confidence: detection.confidence}];
        while (top3.length < 3) top3.push({class_name: '-', confidence: 0, class_id: 0});
        const top3Html = top3.map((c, i) =>
            `<span style="color:${CONFIG.COLORS[c.class_id % CONFIG.COLORS.length]}; font-weight:${i === 0 ? 'bold' : 'normal'}">${c.class_name} ${c.class_name !== '-' ? `(${(c.confidence * 100).toFixed(1)}%)` : ''}</span>`
        ).join('<br>');
        html += `<tr style="border-left: 4px solid ${color};">
            <td style="padding: 8px; border: 1px solid #ddd;">${index + 1}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">${top3Html}</td>
        </tr>`;
    });
    html += '</table>';
    
    // Add species summary
    const speciesCounts = {};
    detections.forEach(detection => {
        speciesCounts[detection.class_name] = (speciesCounts[detection.class_name] || 0) + 1;
    });
    
    html += '<div style="margin-top: 20px;"><h5>🦠 Species Distribution:</h5>';
    html += '<div style="display: flex; flex-wrap: wrap; gap: 10px;">';
    
    Object.entries(speciesCounts).forEach(([species, count]) => {
        const classId = CONFIG.CLASS_NAMES.indexOf(species);
        const color = CONFIG.COLORS[classId % CONFIG.COLORS.length];
        html += `<div style="background: ${color}20; border: 1px solid ${color}; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
            <strong>${species}:</strong> ${count}
        </div>`;
    });
    
    html += '</div></div>';
    detailsList.innerHTML = html;
}

// Show/hide loading indicator
function showLoading(show) {
    const loading = document.getElementById('loadingIndicator');
    loading.style.display = show ? 'block' : 'none';
}

// Debug function for TensorFlow.js YOLO model
function debugTensorFlowJSModel() {
    console.log('=== TensorFlow.js YOLO Model Debug Info ===');
    console.log('Model type:', model.constructor.name);
    console.log('Model signature keys:', Object.keys(model.modelSignature || {}));
    
    // Model inputs
    if (model.inputs) {
        console.log('Model inputs:', model.inputs.map(input => ({
            name: input.name,
            shape: input.shape,
            dtype: input.dtype
        })));
    }
    
    // Model outputs  
    if (model.outputs) {
        console.log('Model outputs:', model.outputs.map(output => ({
            name: output.name,
            shape: output.shape,
            dtype: output.dtype
        })));
    }
    
    console.log('Input size configured:', CONFIG.INPUT_SIZE);
    console.log('Number of classes:', CONFIG.CLASS_NAMES.length);
    console.log('Confidence threshold:', CONFIG.CONFIDENCE_THRESHOLD);
    console.log('Class names:', CONFIG.CLASS_NAMES);
    
    // Test preprocessing with a small test image
    const testCanvas = document.createElement('canvas');
    testCanvas.width = 100;
    testCanvas.height = 100;
    const ctx = testCanvas.getContext('2d');
    
    // Create a simple test pattern
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(0, 0, 50, 50);
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(50, 0, 50, 50);
    ctx.fillStyle = '#0000ff';
    ctx.fillRect(0, 50, 50, 50);
    ctx.fillStyle = '#ffff00';
    ctx.fillRect(50, 50, 50, 50);
    
    console.log('--- Testing Preprocessing ---');
    const testTensor = preprocessImage(testCanvas);
    console.log('Test preprocessing output shape:', testTensor.shape);
    console.log('Expected shape: [1, 640, 640, 3]');
    console.log('Shape matches expected:', 
        testTensor.shape[0] === 1 && 
        testTensor.shape[1] === CONFIG.INPUT_SIZE && 
        testTensor.shape[2] === CONFIG.INPUT_SIZE && 
        testTensor.shape[3] === 3
    );
    
    // Check tensor value range
    const testData = testTensor.dataSync();
    const minVal = Math.min(...testData);
    const maxVal = Math.max(...testData);
    console.log('Tensor value range:', `${minVal.toFixed(3)} to ${maxVal.toFixed(3)}`);
    console.log('Expected range: 0.0 to 1.0');
    testTensor.dispose();
    
    console.log('=== End Debug Info ===');
}

// Enhanced debug function to analyze the exact model output format
async function comprehensiveOutputAnalysis(predictions) {
    console.log('=== COMPREHENSIVE OUTPUT ANALYSIS (NMS Model) ===');
    
    let outputTensor = Array.isArray(predictions) ? predictions[0] : predictions;
    const data = await outputTensor.data();
    const shape = outputTensor.shape;
    
    console.log('Output tensor shape:', shape);
    console.log('Total values in output:', data.length);
    
    // With NMS enabled, output format is typically different
    // Common formats:
    // [1, N, 6] where N is max detections and 6 = [x1, y1, x2, y2, class_id, confidence]
    // [1, N, 7] where 7 = [x1, y1, x2, y2, confidence, class_id, valid]
    
    if (shape.length === 3) {
        if (shape[1] <= 100) {
            console.log('✅ This looks like NMS post-processed output (limited detections)');
            console.log('Format likely: [batch, max_detections, detection_info]');
        } else {
            console.log('⚠️ Large number of detections - NMS might not be applied');
        }
        
        const numDetections = shape[1];
        const outputSize = shape[2];
        
        console.log(`Number of possible detections: ${numDetections}`);
        console.log(`Output size per detection: ${outputSize}`);
        
        // Common NMS output formats
        if (outputSize === 6) {
            console.log('🔍 Likely format: [x1, y1, x2, y2, confidence, class_id]');
        } else if (outputSize === 7) {
            console.log('🔍 Likely format: [x1, y1, x2, y2, confidence, class_id, valid] or similar');
        } else if (outputSize >= 15) {
            console.log('🔍 Likely format: [x_center, y_center, width, height, objectness, class_probs...]');
        } else {
            console.log('🔍 Unknown format - will analyze values');
        }
        
        // Analyze the structure for first few detections
        const numToAnalyze = Math.min(10, numDetections);
        
        console.log(`\nAnalyzing first ${numToAnalyze} detections:`);
        
        for (let i = 0; i < numToAnalyze; i++) {
            const startIdx = i * outputSize;
            const detection = Array.from(data.slice(startIdx, startIdx + outputSize));
            
            console.log(`\nDetection ${i + 1}:`);
            console.log('  Raw values:', detection.map(x => x.toFixed(3)));
            
            // Check if this looks like a valid detection
            const hasNonZeroValues = detection.some(val => Math.abs(val) > 0.001);
            
            if (!hasNonZeroValues) {
                console.log('  Status: Empty/unused detection slot');
                continue;
            }
            
            console.log('  Status: Contains data');
            
            // Try to interpret the format
            if (outputSize >= 6) {
                const first4 = detection.slice(0, 4);
                console.log('  First 4 values (likely bbox):', first4.map(x => x.toFixed(3)));
                
                // Check if coordinates are in [0,1] range (normalized) or pixel values
                const coordsNormalized = first4.every(c => c >= 0 && c <= 1);
                console.log('  Coordinates appear normalized:', coordsNormalized);
                
                if (outputSize === 6) {
                    console.log('  Confidence (likely):', detection[4].toFixed(3));
                    console.log('  Class ID (likely):', detection[5].toFixed(3));
                } else if (outputSize === 7) {
                    console.log('  Value 5:', detection[4].toFixed(3));
                    console.log('  Value 6:', detection[5].toFixed(3));
                    console.log('  Value 7:', detection[6].toFixed(3));
                } else {
                    // Standard YOLO format
                    const objectness = detection[4];
                    const sigmoidObjectness = 1 / (1 + Math.exp(-objectness));
                    console.log('  Objectness raw:', objectness.toFixed(3), '→ sigmoid:', sigmoidObjectness.toFixed(3));
                    
                    const classValues = detection.slice(5, Math.min(15, outputSize));
                    console.log('  Class values (first 10):', classValues.map(x => x.toFixed(3)));
                }
            }
        }
        
        // Look for patterns in the data
        console.log('\n=== DATA PATTERNS ===');
        
        // Count valid detections (non-zero)
        let validDetections = 0;
        let maxConfidence = 0;
        let minConfidence = 1;
        
        for (let i = 0; i < numDetections; i++) {
            const startIdx = i * outputSize;
            const detection = Array.from(data.slice(startIdx, startIdx + outputSize));
            const hasData = detection.some(val => Math.abs(val) > 0.001);
            
            if (hasData) {
                validDetections++;
                
                // Try to extract confidence (different positions based on format)
                let confidence = 0;
                if (outputSize === 6) {
                    confidence = detection[5]; // Last value is typically confidence
                } else if (outputSize === 7) {
                    confidence = Math.max(detection[4], detection[5]); // Could be either position
                } else {
                    // Standard YOLO - calculate from objectness and max class prob
                    const objectness = 1 / (1 + Math.exp(-detection[4]));
                    const classValues = detection.slice(5);
                    const maxClassProb = Math.max(...classValues.map(x => 1 / (1 + Math.exp(-x))));
                    confidence = objectness * maxClassProb;
                }
                
                maxConfidence = Math.max(maxConfidence, confidence);
                minConfidence = Math.min(minConfidence, confidence);
            }
        }
        
        console.log(`Valid detections found: ${validDetections}/${numDetections}`);
        if (validDetections > 0) {
            console.log(`Confidence range: ${minConfidence.toFixed(3)} to ${maxConfidence.toFixed(3)}`);
            
            // Also analyze class distribution
            const classDistribution = {};
            for (let i = 0; i < Math.min(numDetections, 50); i++) {  // Check first 50
                const startIdx = i * outputSize;
                const detection = Array.from(data.slice(startIdx, startIdx + outputSize));
                const hasData = detection.some(val => Math.abs(val) > 0.001);
                
                if (hasData) {
                    const classId = Math.round(detection[5]); // Class is at position 5
                    classDistribution[classId] = (classDistribution[classId] || 0) + 1;
                }
            }
            console.log('Class distribution (first 50 detections):', classDistribution);
        }
        
        // Check for the "always 4 detections" pattern
        if (validDetections === 4) {
            console.log('⚠️ Found exactly 4 detections - this might be the issue!');
            console.log('This suggests the model is outputting a fixed number of top detections');
        }
    }
    
    console.log('=== END COMPREHENSIVE ANALYSIS ===');
}

// Memory monitoring (for debugging)
function logMemoryUsage() {
    const memInfo = tf.memory();
    console.log('🧠 Memory usage:', {
        numTensors: memInfo.numTensors,
        numBytes: memInfo.numBytes,
        unreliable: memInfo.unreliable
    });
    
    if (memInfo.numTensors > 50) {
        console.warn('⚠️ High tensor count detected - possible memory leak');
    }
}

// Error handling wrapper
function handleError(error, context) {
    console.error(`❌ Error in ${context}:`, error);
    
    // Log memory for debugging
    logMemoryUsage();
    
    // Show user-friendly error message
    updateModelStatus(`Error: ${error.message}`, 'error');
}

// Utility function to download results as JSON
function downloadResults(detections, filename = 'bacterial_detection_results.json') {
    const results = {
        timestamp: new Date().toISOString(),
        detections: detections,
        summary: {
            totalColonies: detections.length,
            avgConfidence: detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length,
            totalArea: detections.reduce((sum, d) => sum + d.area, 0)
        }
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Console commands for debugging (call these in browser console)
window.debugFunctions = {
    logMemory: logMemoryUsage,
    downloadResults: downloadResults,
    getModel: () => model,
    getConfig: () => CONFIG,
    debugModel: debugTensorFlowJSModel,
    
    // Comprehensive output analysis
    analyzeOutputs: async () => {
        if (!isModelLoaded) {
            console.log('❌ Model not loaded yet');
            return;
        }
        
        console.log('🔍 Running comprehensive output analysis...');
        
        // Create a test input
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 640;
        testCanvas.height = 640;
        const ctx = testCanvas.getContext('2d');
        
        // Create a realistic test pattern
        ctx.fillStyle = '#2d5a27'; // Dark green (agar plate)
        ctx.fillRect(0, 0, 640, 640);
        
        // Add some white/cream colored "colonies"
        ctx.fillStyle = '#f5f5dc'; // Beige
        ctx.beginPath();
        ctx.arc(150, 150, 12, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(300, 200, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(450, 350, 15, 0, 2 * Math.PI);
        ctx.fill();
        
        try {
            const input = preprocessImage(testCanvas);
            const prediction = await model.executeAsync(input);
            
            // Run the comprehensive analysis
            await comprehensiveOutputAnalysis(prediction);
            
            // Clean up
            if (Array.isArray(prediction)) {
                prediction.forEach(p => p.dispose());
            } else {
                prediction.dispose();
            }
            input.dispose();
            
        } catch (error) {
            console.error('❌ Analysis failed:', error);
        }
    },
    
    // Test full detection pipeline with detailed logging
    testDetectionPipeline: async () => {
        if (!isModelLoaded) {
            console.log('❌ Model not loaded yet');
            return;
        }
        
        console.log('🧪 Testing full detection pipeline...');
        
        // Create test image
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 640;
        testCanvas.height = 640;
        const ctx = testCanvas.getContext('2d');
        
        // Simulate petri dish
        ctx.fillStyle = '#228B22'; // Green agar
        ctx.fillRect(0, 0, 640, 640);
        
        // Add test "bacterial colonies"
        const colonies = [
            {x: 100, y: 100, r: 10, color: '#FFE4B5'}, // Light beige
            {x: 200, y: 150, r: 8, color: '#FFFACD'},  // Light yellow
            {x: 350, y: 200, r: 12, color: '#F0E68C'}, // Khaki
            {x: 450, y: 350, r: 15, color: '#DDA0DD'}, // Plum (different species)
            {x: 500, y: 100, r: 6, color: '#FFB6C1'},  // Light pink
        ];
        
        colonies.forEach(colony => {
            ctx.fillStyle = colony.color;
            ctx.beginPath();
            ctx.arc(colony.x, colony.y, colony.r, 0, 2 * Math.PI);
            ctx.fill();
            
            // Add slight shadow for realism
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            ctx.beginPath();
            ctx.arc(colony.x + 2, colony.y + 2, colony.r, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        console.log(`Created test image with ${colonies.length} simulated colonies`);
        
        try {
            // Run the full pipeline
            const input = preprocessImage(testCanvas);
            const prediction = await model.executeAsync(input);
            const detections = await processDetections(prediction, {
                width: testCanvas.width,
                height: testCanvas.height,
                modelInputSize: CONFIG.INPUT_SIZE
            });
            
            console.log('\n📊 PIPELINE RESULTS:');
            console.log(`Input colonies: ${colonies.length}`);
            console.log(`Detected colonies: ${detections.length}`);
            console.log(`Detection rate: ${((detections.length / colonies.length) * 100).toFixed(1)}%`);
            
            if (detections.length > 0) {
                console.log('\nDetected species:');
                detections.forEach((det, i) => {
                    console.log(`  ${i+1}. ${det.class_name} (${(det.confidence * 100).toFixed(1)}%) at (${Math.round(det.center.x)}, ${Math.round(det.center.y)})`);
                });
            }
            
            // Clean up
            if (Array.isArray(prediction)) {
                prediction.forEach(p => p.dispose());
            } else {
                prediction.dispose();
            }
            input.dispose();
            
        } catch (error) {
            console.error('❌ Pipeline test failed:', error);
        }
    },
    
    // Check class mapping
    checkClassMapping: async () => {
        if (!isModelLoaded) {
            console.log('❌ Model not loaded yet');
            return;
        }
        
        console.log('🔍 Checking class mapping...');
        console.log('Configured classes:', CONFIG.CLASS_NAMES);
        console.log('Number of classes:', CONFIG.CLASS_NAMES.length);
        
        // Test with a simple input to see what classes are predicted
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 640;
        testCanvas.height = 640;
        const ctx = testCanvas.getContext('2d');
        ctx.fillStyle = '#008000';
        ctx.fillRect(0, 0, 640, 640);
        
        // Add one white circle
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(320, 320, 20, 0, 2 * Math.PI);
        ctx.fill();
        
        try {
            const input = preprocessImage(testCanvas);
            const prediction = await model.executeAsync(input);
            const outputTensor = Array.isArray(prediction) ? prediction[0] : prediction;
            const data = await outputTensor.data();
            
            // Look at class predictions for first few detections
            const numDetections = outputTensor.shape[1];
            const outputSize = outputTensor.shape[2];
            const numClasses = outputSize - 5;
            
            console.log(`Model expects ${numClasses} classes`);
            console.log(`Config has ${CONFIG.CLASS_NAMES.length} classes`);
            
            if (numClasses !== CONFIG.CLASS_NAMES.length) {
                console.error('❌ CLASS MISMATCH! This explains wrong predictions.');
                console.log('Possible solutions:');
                console.log('1. Update CONFIG.CLASS_NAMES to match model output');
                console.log('2. Re-export model with correct number of classes');
                console.log('3. Check if model includes background class');
            }
            
            // Show raw class values for first detection
            if (numDetections > 0) {
                const firstDetectionStart = 0 * outputSize;
                console.log('\nFirst detection raw class values:');
                for (let c = 0; c < Math.min(numClasses, 15); c++) {
                    const rawValue = data[firstDetectionStart + 5 + c];
                    const sigmoid = 1 / (1 + Math.exp(-rawValue));
                    console.log(`  Class ${c}: ${rawValue.toFixed(3)} → ${(sigmoid * 100).toFixed(1)}%`);
                }
            }
            
            // Clean up
            if (Array.isArray(prediction)) {
                prediction.forEach(p => p.dispose());
            } else {
                prediction.dispose();
            }
            input.dispose();
            
        } catch (error) {
            console.error('❌ Class mapping check failed:', error);
        }
    },
    
    // Test sigmoid conversion
    testSigmoid: () => {
        console.log('Testing sigmoid conversion:');
        const testLogits = [-10, -5, -1, 0, 1, 5, 10, 20];
        testLogits.forEach(logit => {
            const sigmoid = 1 / (1 + Math.exp(-logit));
            console.log(`Logit ${logit.toString().padStart(3)} → Sigmoid ${sigmoid.toFixed(4)} (${(sigmoid * 100).toFixed(2)}%)`);
        });
    },
    
    // Test preprocessing
    testPreprocessing: () => {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(0, 0, 640, 640);
        const tensor = preprocessImage(canvas);
        console.log('Test tensor shape:', tensor.shape);
        console.log('Test tensor range:', [tensor.min().dataSync()[0], tensor.max().dataSync()[0]]);
        tensor.dispose();
    },
    
    // Test coordinate scaling with different image sizes
    testCoordinateScaling: () => {
        console.log('Testing coordinate scaling with different image sizes:');
        
        const testSizes = [
            { width: 320, height: 240, name: '320x240 (4:3)' },
            { width: 640, height: 640, name: '640x640 (square)' },
            { width: 800, height: 600, name: '800x600 (4:3)' },
            { width: 1024, height: 768, name: '1024x768 (4:3)' },
            { width: 1920, height: 1080, name: '1920x1080 (16:9)' }
        ];
        
        testSizes.forEach(size => {
            // Simulate model coordinates (640x640 space)
            const modelCoords = { x1: 100, y1: 150, x2: 200, y2: 250 };
            
            // Scale to original size
            const scaleX = size.width / CONFIG.INPUT_SIZE;
            const scaleY = size.height / CONFIG.INPUT_SIZE;
            const scaledCoords = {
                x1: modelCoords.x1 * scaleX,
                y1: modelCoords.y1 * scaleY,
                x2: modelCoords.x2 * scaleX,
                y2: modelCoords.y2 * scaleY
            };
            
            console.log(`${size.name}:`);
            console.log(`  Scale factors: x=${scaleX.toFixed(3)}, y=${scaleY.toFixed(3)}`);
            console.log(`  Model coords: [${modelCoords.x1}, ${modelCoords.y1}, ${modelCoords.x2}, ${modelCoords.y2}]`);
            console.log(`  Scaled coords: [${scaledCoords.x1.toFixed(1)}, ${scaledCoords.y1.toFixed(1)}, ${scaledCoords.x2.toFixed(1)}, ${scaledCoords.y2.toFixed(1)}]`);
        });
    },
    
    // Test hover functionality
    testHover: () => {
        const canvas = document.getElementById('detectionCanvas');
        if (!canvas) {
            console.log('❌ Canvas not found - upload an image first');
            return;
        }
        
        console.log('🖱️ Testing hover functionality:');
        console.log('Canvas element:', canvas);
        console.log('Canvas style.pointerEvents:', canvas.style.pointerEvents);
        console.log('Canvas detection data:', canvas.detectionData?.length || 0, 'detections');
        
        if (canvas.detectionData && canvas.detectionData.length > 0) {
            console.log('First detection display box:', canvas.detectionData[0].displayBox);
            console.log('Canvas display size:', canvas.offsetWidth, 'x', canvas.offsetHeight);
            console.log('Try moving your mouse over the bounding boxes');
        } else {
            console.log('❌ No detection data found - process an image first');
        }
        
        // Test if event listeners are attached
        console.log('Hover handler attached:', !!canvas._hoverHandler);
        console.log('Leave handler attached:', !!canvas._leaveHandler);
    }
};