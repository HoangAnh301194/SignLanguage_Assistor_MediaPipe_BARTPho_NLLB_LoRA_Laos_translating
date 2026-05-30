import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.tflite"
LABEL_MAP_PATH = ROOT / "sign_to_prediction_index_map.json"


class ASL250Recognizer:
    def __init__(
        self,
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        camera=None,
        max_frames=60,
        min_frames=16,
        predict_every=8,
        top_k=3,
    ):
        self.camera = camera
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.predict_every = predict_every
        self.top_k = top_k

        with open(label_map_path, "r", encoding="utf-8") as f:
            sign_to_id = json.load(f)
        self.id_to_sign = {idx: sign for sign, idx in sign_to_id.items()}

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.frames = deque(maxlen=max_frames)
        self.frame_counter = 0
        self.last_top = []

    def _camera_candidates(self):
        candidates = []
        if self.camera is not None:
            candidates.append(self.camera)
        candidates.extend([0, 1, 2, 3, 4])
        return list(dict.fromkeys(candidates))

    def _open_camera(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for camera in self._camera_candidates():
            for backend in backends:
                cap = cv2.VideoCapture(camera, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera opened successfully at index {camera}")
                        return cap
                cap.release()
        return None

    @staticmethod
    def _landmarks_to_array(landmarks, count):
        if landmarks is None:
            return np.full((count, 3), np.nan, dtype=np.float32)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    def _extract_frame_landmarks(self, results):
        # Kaggle ASL Signs row order: face(468), left_hand(21), pose(33), right_hand(21)
        face = self._landmarks_to_array(results.face_landmarks, 468)
        left = self._landmarks_to_array(results.left_hand_landmarks, 21)
        pose = self._landmarks_to_array(results.pose_landmarks, 33)
        right = self._landmarks_to_array(results.right_hand_landmarks, 21)
        return np.concatenate([face, left, pose, right], axis=0)

    def _hands_visible(self, results):
        return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

    def _softmax_if_needed(self, scores):
        scores = scores.astype(np.float32)
        total = float(np.sum(scores))
        if np.all(scores >= 0) and 0.95 <= total <= 1.05:
            return scores
        exp = np.exp(scores - np.max(scores))
        return exp / np.sum(exp)

    def predict(self):
        if len(self.frames) < self.min_frames:
            return []

        inputs = np.array(self.frames, dtype=np.float32)
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape, strict=False)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(self.output_index)
        scores = self._softmax_if_needed(scores)
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        return [(self.id_to_sign[int(i)], float(scores[int(i)])) for i in top_indices]

    def _draw_results(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        cv2.rectangle(frame, (0, 0), (520, 142), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"ASL 250 | frames: {len(self.frames)}/{self.max_frames}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        if self.last_top:
            for row, (label, score) in enumerate(self.last_top):
                cv2.putText(
                    frame,
                    f"{row + 1}. {label}: {score * 100:.1f}%",
                    (12, 62 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if row == 0 else (220, 220, 220),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "Show a sign. Press SPACE to reset.",
                (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 200, 255),
                2,
            )
        return frame

    def run(self):
        cap = self._open_camera()
        if cap is None:
            print("ERROR: Could not open a working camera.")
            return

        print("Starting ASL 250 realtime demo.")
        print("Press SPACE to reset, q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.holistic.process(rgb)
            rgb.flags.writeable = True

            if self._hands_visible(results):
                self.frames.append(self._extract_frame_landmarks(results))
                self.frame_counter += 1
                if self.frame_counter % self.predict_every == 0:
                    self.last_top = self.predict()
                    if self.last_top:
                        print(
                            "\r"
                            + " | ".join(
                                f"{label}: {score * 100:.1f}%"
                                for label, score in self.last_top
                            ),
                            end="",
                            flush=True,
                        )

            frame = self._draw_results(frame, results)
            cv2.imshow("ASL 250 Realtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                self.frames.clear()
                self.last_top = []
                self.frame_counter = 0

        print("\nClosing...")
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()


def main():
    parser = argparse.ArgumentParser(description="Realtime ASL 250-sign demo")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--min-frames", type=int, default=16)
    args = parser.parse_args()

    recognizer = ASL250Recognizer(
        camera=args.camera,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
    )
    recognizer.run()


if __name__ == "__main__":
    main()

