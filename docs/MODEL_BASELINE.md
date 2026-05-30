# Baseline Recognition Model Documentation

## Overview

The baseline gesture recognition model uses a pre-trained ASL-250 recognition model from Google, adapted for Vietnamese Sign Language recognition through an intermediate token mapping layer.

## Model Information

### Google ASL-250 Model

**Publication**: "Towards Accurate Sign Language Recognition with Semantic Alignment"

**Model Type**: TensorFlow Lite quantized model

**File Location**: `external/asl_baseline/model.tflite`

**Model Size**: ~11 MB

**Supported Gestures**: 250 American Sign Language (ASL) classes

### Architecture Components

1. **Input Processing**
   - MediaPipe Holistic for pose estimation
   - Extracts hand keypoints (21 points per hand)
   - Extracts body keypoints (33 points)
   - Face landmarks (468 points, optional)

2. **Recognition Network**
   - Lightweight TensorFlow Lite model
   - Optimized for real-time mobile inference
   - Quantized for faster computation

3. **Output**
   - Class predictions (0-249)
   - Confidence scores (0-1)
   - Requires post-processing for temporal smoothing

## Model Configuration

### Input Specifications
```python
Input Shape: (1, sequence_length, num_keypoints)
Data Type: float32
Sequence Length: Variable (typically 10-30 frames)
Keypoint Dimensions: 3 (x, y, confidence)
```

### Output Specifications
```python
Output Shape: (1, 250)
Data Type: float32
Interpretation: Probability distribution over gesture classes
```

## Evaluation Results

### Original ASL-250 Benchmark
- **Accuracy**: 94.5% on test set
- **Latency**: ~30ms per frame (NVIDIA GPU)
- **Mobile Latency**: ~100ms per frame (mobile GPU)
- **Model Size**: ~11 MB

### Vietnamese Sign Language Adaptation
- **Recognition Accuracy**: ~85-90% (adapted to VSL)
- **Performance Degradation**: Due to ASL-VSL differences
- **Latency**: Maintained (gesture recognition unchanged)

### Per-Class Performance
See `external/asl_baseline/eval_results/recognition_latency_benchmark.json` for detailed metrics:

```json
{
  "average_latency_ms": 45.2,
  "min_latency_ms": 28.1,
  "max_latency_ms": 156.3,
  "per_class_accuracy": {...}
}
```

## Usage

### Direct Model Inference

```python
from external.google_asl_250 import realtime_asl_250

recognizer = realtime_asl_250.GestureRecognizer(
    model_path="external/asl_baseline/model.tflite",
    threshold=0.5
)

# Process frame
landmarks = extract_landmarks_from_frame(frame)
gesture_class, confidence = recognizer.predict(landmarks)
```

### Real-time Recognition

```bash
python external/asl_baseline/realtime_asl_250.py \
    --model external/asl_baseline/model.tflite \
    --device 0  # Webcam ID
```

### Evaluation

```bash
python external/asl_baseline/evaluate_asl_250.py \
    --model external/asl_baseline/model.tflite \
    --test-data path/to/test/data \
    --output results.json
```

## Gesture-to-Token Mapping

**File**: `external/asl_baseline/sign_to_prediction_index_map.json`

**Purpose**: Maps ASL-250 class indices to:
- ASL gesture names
- Equivalent Vietnamese gesture labels
- Vietnamese NLP tokens

**Example**:
```json
{
  "0": {
    "asl_name": "hello",
    "vsl_name": "xin_chào",
    "token": "xin_chao"
  },
  "1": {
    "asl_name": "goodbye",
    "vsl_name": "tạm_biệt",
    "token": "tam_biet"
  }
}
```

This mapping allows the baseline ASL model to work with Vietnamese-specific training data through the integrated pipeline.

## Performance Optimization

### Quantization
- Model is already int8 quantized for TensorFlow Lite
- No further quantization needed
- Trade-off: Small accuracy loss for significant speed improvement

### Batch Processing
Not applicable for real-time stream processing, but available for offline analysis:

```python
# Process batch of frames
predictions = recognizer.predict_batch(landmarks_batch)
```

### GPU Acceleration

**TensorFlow Lite GPU Delegate**:
```python
# Enable GPU acceleration (requires GPU support)
recognizer = GestureRecognizer(
    model_path="model.tflite",
    use_gpu=True
)
```

## Limitations & Known Issues

### Accuracy Limitations
1. **ASL-VSL Differences**: Model trained on ASL, may not recognize VSL-specific gestures
2. **View Dependency**: Performs best with frontal view (0-30° angle)
3. **Lighting Sensitivity**: Performance degrades in low-light conditions
4. **Speed Dependency**: Fast movements may be misrecognized
5. **Occlusion**: Hands/body partially covered reduces accuracy

### Environmental Constraints
- Requires clear background (works less well with complex backgrounds)
- Sensitive to hand size variations
- Single-hand gestures recognized better than two-handed
- Requires good video quality (720p minimum recommended)

### Model Constraints
- Fixed gesture set (250 classes only)
- No gesture-to-word mapping included
- Requires external text generation module for full understanding
- Cannot recognize fingerspelling (letter-by-letter spelling)

## Future Improvements

### Short-term
1. Fine-tune model on Vietnamese Sign Language data
2. Increase gesture recognition to 400+ classes
3. Implement temporal smoothing for stability
4. Add confidence-based filtering

### Long-term
1. Develop VSL-native end-to-end model
2. Support multi-hand gesture recognition
3. Add fingerspelling recognition module
4. Implement gesture-to-word direct mapping
5. Real-time performance on edge devices (mobile, embedded)

## References

**Paper**: "Towards Accurate Sign Language Recognition with Semantic Alignment"

**MediaPipe Holistic**: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/

**TensorFlow Lite**: https://www.tensorflow.org/lite

**Dataset**: VSL400 - Vietnamese Sign Language dataset for word-level recognition

## Troubleshooting

### Issue: Model File Not Found
```bash
# Ensure model file exists
ls -la external/asl_baseline/model.tflite
```

### Issue: Poor Recognition Accuracy
- Increase lighting in environment
- Maintain clear hand visibility
- Keep frontal body position
- Move slowly and deliberately
- Increase confidence threshold

### Issue: High Latency
- Enable GPU acceleration
- Use NNAPI or CoreML delegates
- Process lower resolution input
- Skip non-consecutive frames

### Issue: Out of Memory
- Reduce batch size
- Process shorter sequences
- Use model quantization (already applied)

## Model Files

```
external/asl_baseline/
├── model.tflite                           # Model weights (11 MB)
├── sign_to_prediction_index_map.json      # Gesture mappings
├── realtime_asl_250.py                    # Real-time inference script
├── evaluate_asl_250.py                    # Evaluation script
└── eval_results/
    └── recognition_latency_benchmark.json # Performance metrics
```

## For More Information

- See [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) for system architecture
- Check [INTEGRATION_PIPELINE.md](./INTEGRATION_PIPELINE.md) for pipeline details
- Refer to module READMEs for component-specific information

