README – Dataset for "A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition"
==================================================================

Created on: 2025-12-15
Created by: Trung Nguyen Quoc, Researcher (VSB-TUO, quoc.trung.nguyen.st@vsb.cz)
Last update: 2025-12-15

------------------------------------------------------------------
Basic information
------------------------------------------------------------------

1. Journal article: A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition

2. DOI:10.5281/zenodo.17943574.

3. Contact information:
   Name: Matej Sindelar
   Institution: VSB – Technical University of Ostrava
   E-mail: matej.sindelar@vsb.cz
   ORCID: https://orcid.org/0009-0006-0556-507X

4. Dataset publication date: 2026-03-09

5. Place of publication: Ostrava, Czechia

6. Dataset Description
================================================================================

# VSL400 Blurred Splits Dataset

This directory contains the VSL400 dataset split into 7 synchronized splits. Each split contains synchronized videos from three camera views (front, left, right) with corresponding metadata JSONs.

## Dataset Structure

```
vsl400_blurred_splits/
├── split_1/
│   ├── front_view.json
│   ├── left_view.json
│   ├── right_view.json
│   ├── front_view/          (video files)
│   ├── left_view/           (video files)
│   └── right_view/          (video files)
├── split_2/
│   ├── front_view.json
│   ├── left_view.json
│   ├── right_view.json
│   ├── front_view/
│   ├── left_view/
│   └── right_view/
├── split_3/
├── split_4/
├── split_5/
├── split_6/
├── split_7/
└── README.md (this file)
```

## Split Details

Each split contains synchronized video IDs across all three views. The splits are designed for balanced dataset distribution across training, validation, and testing phases.

### Split 1
- **Video ID Range:** 000000 - 003540
- **Number of Videos per View:** 3,541
- **Total Videos:** 10,623 (3,541 × 3 views)
- **Description:** Contains the beginning of the dataset with signers starting from ID 001

### Split 2
- **Video ID Range:** 003541 - 007081
- **Number of Videos per View:** 3,541
- **Total Videos:** 10,623 (3,541 × 3 views)
- **Description:** Contains videos from signers ID 006 onwards

### Split 3
- **Video ID Range:** 007082 - 010617
- **Number of Videos per View:** 3,536
- **Total Videos:** 10,608 (3,536 × 3 views)
- **Description:** Contains videos from signers ID 010 onwards

### Split 4
- **Video ID Range:** 010618 - 014158
- **Number of Videos per View:** 3,541
- **Total Videos:** 10,623 (3,541 × 3 views)
- **Description:** Contains videos from signers ID 014 onwards

### Split 5
- **Video ID Range:** 014159 - 017699
- **Number of Videos per View:** 3,541
- **Total Videos:** 10,623 (3,541 × 3 views)
- **Description:** Contains videos from signers ID 018 onwards

### Split 6
- **Video ID Range:** 017700 - 021238
- **Number of Videos per View:** 3,539
- **Total Videos:** 10,617 (3,539 × 3 views)
- **Description:** Contains videos from signers ID 022 onwards

### Split 7
- **Video ID Range:** 021239 - 024774
- **Number of Videos per View:** 3,536
- **Total Videos:** 10,608 (3,536 × 3 views)
- **Description:** Contains the final videos in the dataset from signers ID 024 onwards

## Metadata Files

Each split includes three JSON files (`front_view.json`, `left_view.json`, `right_view.json`) containing metadata for the videos in that split. Each JSON is a list of objects with the following structure:

```json
{
  "video_id": "000000",
  "signer_id": "001",
  "fps": 25.0,
  "resolution": 1080,
  "gloss": "Anh",
  "num_frames": 65,
  "length_seconds": 2.6
}
```

### Metadata Fields
- **video_id**: Unique identifier for the video (zero-padded 6-digit string)
- **signer_id**: Identifier for the signer performing the gesture (zero-padded 3-digit string)
- **fps**: Frames per second of the video (25.0)
- **resolution**: Video resolution in pixels (1080 for 1080p)
- **gloss**: English gloss/translation of the sign language gesture
- **num_frames**: Total number of frames in the video
- **length_seconds**: Duration of the video in seconds

## Video Files

Each view directory contains synchronized video files corresponding to the IDs listed in the metadata JSON. Files are named with the zero-padded video ID followed by the file extension (e.g., `000000.mp4`).

### Synchronized Videos
All three views (front_view, left_view, right_view) are synchronized and captured simultaneously. For any given video_id in the split, corresponding videos should exist in all three view directories with identical content (same signer, same gesture) recorded from different angles.

## Usage

### Loading Data from a Single Split

```python
import json
from pathlib import Path

split_dir = Path('vsl400_blurred_splits/split_1')

# Load metadata
with open(split_dir / 'front_view.json') as f:
    front_metadata = json.load(f)

with open(split_dir / 'left_view.json') as f:
    left_metadata = json.load(f)

with open(split_dir / 'right_view.json') as f:
    right_metadata = json.load(f)

# Access video files
front_videos = list((split_dir / 'front_view').glob('*.mp4'))
left_videos = list((split_dir / 'left_view').glob('*.mp4'))
right_videos = list((split_dir / 'right_view').glob('*.mp4'))
```

### Loading Data from All Splits

```python
import json
from pathlib import Path

all_metadata = {'front': [], 'left': [], 'right': []}

for split_idx in range(1, 8):
    split_dir = Path(f'vsl400_blurred_splits/split_{split_idx}')
    
    with open(split_dir / 'front_view.json') as f:
        all_metadata['front'].extend(json.load(f))
    
    with open(split_dir / 'left_view.json') as f:
        all_metadata['left'].extend(json.load(f))
    
    with open(split_dir / 'right_view.json') as f:
        all_metadata['right'].extend(json.load(f))
```

## Dataset Statistics

| Split | Video ID Range | Videos per View | Total Videos |
|-------|-----------------|-----------------|--------------|
| 1     | 000000 - 003540 | 3,541           | 10,623       |
| 2     | 003541 - 007081 | 3,541           | 10,623       |
| 3     | 007082 - 010617 | 3,536           | 10,608       |
| 4     | 010618 - 014158 | 3,541           | 10,623       |
| 5     | 014159 - 017699 | 3,541           | 10,623       |
| 6     | 017700 - 021238 | 3,539           | 10,617       |
| 7     | 021239 - 024774 | 3,536           | 10,608       |
| **Total** | 000000 - 024774 | **24,775** | **74,325** |

## Merging All Splits Back Together

Use `merge_splits.py` (in this directory) to combine all `split_*` folders into a single merged dataset with unified JSONs and videos.

### Quick merge (copies files)
```cmd
python merge_splits.py --splits-root . --out merged
```

### Faster merge with hardlinks (same drive/volume)
```cmd
python merge_splits.py --splits-root . --out merged --copy-mode hardlink
```

### Options
- `--splits-root`: directory containing `split_1..split_7` (default `.`)
- `--out`: output folder for merged data (default `merged`)
- `--copy-mode`: `copy` (default), `hardlink`, or `symlink` (falls back to copy if not permitted)
- `--overwrite`: allow overwriting existing files in the output
- `--stop-on-dup`: fail if duplicate `video_id` is encountered across splits
- `--verbose`: verbose logging

### What the script does
- Merges JSONs per view into `merged/front_view.json`, `merged/left_view.json`, `merged/right_view.json`.
- Merges videos per view into `merged/front_view/`, `merged/left_view/`, `merged/right_view/`.
- Skips non-video files; accepts common video extensions (.mp4, .avi, .mov, .mkv, .webm).

## Processing Notes

- All videos are pre-processed with **face blurring applied** for privacy protection. Only the upper half of detected faces (nose and above) are blurred.
- All videos have been **downsampled to 25 fps** from the original higher fps.
- Video resolution is standardized at **1080p**.
- All three views are **synchronized** (same signer, same gesture, different camera angles).
- Metadata is provided in **JSON format** with synchronized entries across all three view JSONs per split.

## Notes

- The splits are designed to be balanced in size for fair distribution across machine learning train/validation/test phases.
- Video IDs are **not contiguous per signer**—a single signer may have videos scattered across different splits.
- Each split's JSON files contain **filtered entries only for videos present in that split's directories**.
- Videos are **face-blurred** with privacy constraints; facial recognition is not possible from these videos.

---
# VietnameseSignLanguageRecognition - SourceCode

This source code encompasses the processing of raw video and models; further details are available in the readme file within the code directory.



For questions or issues regarding the dataset structure or content, please refer to the main project documentation or contact the dataset maintainers.

