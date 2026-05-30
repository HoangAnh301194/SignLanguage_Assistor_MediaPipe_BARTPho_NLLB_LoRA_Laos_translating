# Integration Pipeline Guide

This document describes the complete integration pipeline that connects all system components.

## Pipeline Architecture

The integrated pipeline combines three main components:

1. **ASL Recognition Module** (`external/asl_baseline/`)
   - Real-time gesture recognition using MediaPipe Holistic + TensorFlow Lite
   - Outputs gesture class predictions with confidence scores

2. **Vietnamese NLP Module** (`VSL_Vietnamese_NLP/`)
   - Converts gesture sequences into Vietnamese tokens
   - Generates fluent Vietnamese sentences using BART-PHO

3. **Vi-Lao Translation Module** (`dich-vi_lo/`)
   - Translates Vietnamese sentences to Lao
   - Uses NLLB model with LoRA fine-tuning

## Gesture-to-Token Mapping

The integration layer maintains an ASL-to-Vietnamese token mapping (`ASL_TO_VI_TOKEN` in `integrated_pipeline.py`) because:
- Current recognizer is ASL-based (250 classes)
- VSL Vietnamese dataset uses different gesture labels
- Mapping provides adapter layer for future VSL-native models

## Running the Integrated Pipeline

### 1. Batch Processing (Video File)

**Without Translation (Smoke Test):**
```bash
python scripts/integrated_pipeline.py --no-translation
```

Example input:
```text
hello
/build
```

Expected output:
```text
Token sequence: xin_chao
Vietnamese: Xin chào.
```

**With Full Translation (Vietnamese → Lao):**
```bash
python scripts/integrated_pipeline.py
```

First run will download the base model `facebook/nllb-200-distilled-600M`, requiring:
- Internet connection
- ~1-2 GB disk space
- Several minutes download time

### 2. Real-time Webcam Processing

**Without Translation:**
```bash
python scripts/realtime_integrated.py --no-translation
```

**With Translation:**
```bash
python scripts/realtime_integrated.py
```

### Keyboard Controls

During real-time processing:

| Key | Action |
|-----|--------|
| `a` | Accept current top-1 prediction into gesture buffer |
| `b` | Build gesture buffer into Vietnamese sentence (translate if enabled) |
| `c` | Clear gesture buffer |
| `SPACE` | Reset gesture recognition frame buffer |
| `q` | Quit application |

### Example Real-time Workflow

1. User signs a sequence
2. Press `a` to collect gestures
3. Press `b` to generate Vietnamese sentence and Lao translation
4. Results displayed on screen
5. Press `c` to clear and start over

## Configuration

### Pipeline Parameters

Edit `scripts/integrated_pipeline.py` to adjust:

- `CONFIDENCE_THRESHOLD`: Minimum prediction confidence (default: 0.5)
- `FRAME_BUFFER_SIZE`: Frames to accumulate before processing (default: 5)
- `NLP_BATCH_SIZE`: Batch size for NLP model (default: 1)

### Model Selection

Translation model options in `dich-vi_lo/translate.py`:

```python
# Lightweight (faster)
model_name = "facebook/nllb-200-distilled-600M"

# Full model (better quality)
model_name = "facebook/nllb-200-3.3B"
```

## Performance Optimization

### GPU Acceleration

Enable CUDA for faster processing:
```bash
# Will auto-detect CUDA if available
python scripts/realtime_integrated.py --gpu
```

### Quantization

For deployment on resource-limited devices:
```bash
# Run with quantized models
python scripts/realtime_integrated.py --quantize
```

### Frame Skipping

Process every N frames to reduce latency:
```python
# In realtime_integrated.py
FRAME_SKIP = 2  # Process every 2nd frame
```

## Output Formats

### Batch Processing Output

```json
{
  "metadata": {
    "video": "input.mp4",
    "frames_processed": 240,
    "timestamp": "2026-05-29T10:00:00"
  },
  "results": [
    {
      "frame_id": 0,
      "gesture": "hello",
      "confidence": 0.95,
      "vietnamese": "Xin chào",
      "lao": "ສະຫວັດດີ"
    }
  ]
}
```

### Real-time Display Output

```
Frame: 0 | Gesture: hello (0.95) | Vietnamese: Xin chào | Lao: ສະຫວັດດີ
Frame: 1 | Gesture: world (0.87) | Vietnamese: Thế giới | Lao: ໂລກ
...
```

## Troubleshooting

### Issue: Low Recognition Confidence

**Solution**: Increase lighting, maintain frontal position, move slowly

### Issue: Model Not Found

**Solution**: Ensure models are downloaded and in correct directories:
```bash
# Check model paths
ls VSL_Vietnamese_NLP/models/
ls dich-vi_lo/model/
ls external/asl_baseline/
```

### Issue: Out of Memory

**Solution**: 
- Reduce frame buffer size
- Enable quantization
- Process video in shorter chunks
- Reduce batch size

### Issue: Slow Performance

**Solution**:
- Enable GPU acceleration
- Enable frame skipping
- Use distilled models
- Check CPU utilization

## Advanced Usage

### Custom Gesture Mapping

To use different gesture labels:

1. Edit `ASL_TO_VI_TOKEN` in `integrated_pipeline.py`
2. Map ASL class indices to your token labels
3. Ensure tokens exist in `VSL_Vietnamese_NLP` vocabulary

### Batch Processing Multiple Videos

```bash
for video in videos/*.mp4; do
    python scripts/integrated_pipeline.py --input "$video" \
                                          --output "results/$(basename $video .mp4).json"
done
```

### Export to Different Formats

```bash
# Output as CSV
python scripts/integrated_pipeline.py --output results.csv --format csv

# Output as XML
python scripts/integrated_pipeline.py --output results.xml --format xml
```

## Performance Metrics

Typical latencies (with GPU):
- ASL Recognition: 50-100ms per frame
- Sentence Generation: 200-500ms per sequence
- Translation: 100-200ms per sentence
- **Total pipeline**: <2 seconds per complete sentence

## Component Integration Details

### Data Flow

```
Video Input
    ↓
[Frame Processing]
    ↓
[Gesture Recognition] → ASL confidence scores
    ↓
[Token Mapping] → Vietnamese tokens
    ↓
[Sentence Builder] → Vietnamese sentence
    ↓
[Translator] → Lao translation
    ↓
Text Output (Vietnamese + Lao)
```

### Error Handling

The pipeline implements:
- Confidence threshold filtering
- Graceful degradation when models unavailable
- Logging of skipped or failed frames
- Fallback modes for missing components

## Next Steps

- See [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) for system details
- Refer to module-specific READMEs for component tuning
- Check [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines

