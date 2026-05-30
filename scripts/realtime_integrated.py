import argparse
import importlib.util
import sys
import time
import unicodedata
import threading
from collections import Counter, deque
from pathlib import Path

import cv2

from integrated_pipeline import SignLanguageTranslationPipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
ASL_SCRIPT = ROOT / "external" / "google-asl-250" / "realtime_asl_250.py"


def load_asl_recognizer_class():
    spec = importlib.util.spec_from_file_location("realtime_asl_250", ASL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ASL250Recognizer


class IntegratedRealtimeRecognizer:
    def __init__(
        self,
        camera=None,
        confidence_threshold=0.55,
        accept_cooldown=1.2,
        enable_translation=True,
        use_bartpho=False,
        auto_accept=False,
        stable_predictions=3,
        stable_window=5,
        fullscreen=True,
    ):
        recognizer_cls = load_asl_recognizer_class()
        self.recognizer = recognizer_cls(camera=camera)
        self.pipeline = SignLanguageTranslationPipeline(
            use_bartpho=use_bartpho,
            enable_translation=enable_translation,
        )
        self.confidence_threshold = confidence_threshold
        self.accept_cooldown = accept_cooldown
        self.enable_translation = enable_translation
        self.auto_accept = auto_accept
        self.stable_predictions = stable_predictions
        self.prediction_history = deque(maxlen=stable_window)
        self.last_accept_time = 0.0
        self.last_sign = None
        self.last_result = {"raw": "", "vietnamese": "", "lao": ""}
        self.status_message = "Waiting for a stable sign."
        self.fullscreen = fullscreen
        
        # Screen scale factor for responsive UI
        self.screen_width = 1920
        self.screen_height = 1080
        self.is_building = False
        self.scale_factor = 1.0
        self._update_scale_factor()

    def _update_scale_factor(self):
        """Calculate scale factor based on screen resolution"""
        # Reference resolution is 1920x1080
        # Adjust based on actual screen size
        width_ratio = self.screen_width / 1920
        height_ratio = self.screen_height / 1080
        self.scale_factor = min(width_ratio, height_ratio)
        # Clamp scale factor to reasonable range
        self.scale_factor = max(0.5, min(2.0, self.scale_factor))
        print(f"[UI] Scale factor: {self.scale_factor:.2f} (resolution: {self.screen_width}x{self.screen_height})")

    def _accept_top_prediction(self):
        if not self.recognizer.last_top:
            self.status_message = "No prediction to accept."
            return

        sign, score = self.recognizer.last_top[0]
        now = time.time()
        if score < self.confidence_threshold:
            self.status_message = (
                f"Rejected {sign}: {score * 100:.1f}% < {self.confidence_threshold * 100:.0f}%"
            )
            return
        if sign == self.last_sign and now - self.last_accept_time < self.accept_cooldown:
            self.status_message = f"Skipped duplicate: {sign}"
            return

        raw_before = self.pipeline.buffer.raw_sentence()
        token = self.pipeline.add_sign(sign)
        raw_after = self.pipeline.buffer.raw_sentence()
        self.last_sign = sign
        self.last_accept_time = now

        if raw_after == raw_before:
            self.status_message = f"Skipped by buffer cooldown: {sign}"
        else:
            self.status_message = f"Accepted: {sign} -> {token} ({score * 100:.1f}%)"
            print(f"\n{self.status_message}")

    def _update_stability(self):
        if not self.recognizer.last_top:
            return

        sign, score = self.recognizer.last_top[0]
        if score < self.confidence_threshold:
            self.prediction_history.clear()
            return

        self.prediction_history.append(sign)

        if self.auto_accept and len(self.prediction_history) >= self.stable_predictions:
            sign_counts = Counter(self.prediction_history)
            stable_sign, stable_count = sign_counts.most_common(1)[0]
            if stable_sign == sign and stable_count >= self.stable_predictions:
                self._accept_top_prediction()
                self.prediction_history.clear()

    def _put_text(self, frame, text, origin, scale=0.58, color=(255, 255, 255), thickness=1):
        """Put text on frame with auto-scaled font size"""
        scaled_font_size = scale  # Keep original scale, don't multiply by scale_factor
        scaled_thickness = max(1, int(thickness))  # Keep original thickness
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scaled_font_size,
            color,
            scaled_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _ascii_text(text):
        normalized = unicodedata.normalize("NFD", text or "")
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").encode(
            "ascii", "ignore"
        ).decode("ascii")

    @staticmethod
    def _blend_panel(frame, p1, p2, color=(16, 20, 28), alpha=0.78):
        overlay = frame.copy()
        cv2.rectangle(overlay, p1, p2, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_landmarks(self, frame, results):
        recognizer = self.recognizer
        if results.pose_landmarks:
            recognizer.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                recognizer.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=recognizer.mp_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            recognizer.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                recognizer.mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            recognizer.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                recognizer.mp_holistic.HAND_CONNECTIONS,
            )
        return frame

    def _draw_prediction_panel(self, frame):
        h, w = frame.shape[:2]
        
        # Panel dimensions - Reduced size
        padding = int(8 * self.scale_factor)
        panel_w = int(min(260, w - 16) * self.scale_factor)
        panel_h = int(110 * self.scale_factor)
        line_height = int(22 * self.scale_factor)
        
        self._blend_panel(frame, (padding, padding), (padding + panel_w, padding + panel_h))
        cv2.rectangle(frame, (padding, padding), (padding + panel_w, padding + panel_h), (70, 92, 120), 1)

        mode = "AUTO" if self.auto_accept else "MANUAL"
        self._put_text(frame, "ASL-250", (padding + 8, padding + 22), 0.55, (255, 255, 255), 2)
        self._put_text(
            frame,
            f"{mode} | frames {len(self.recognizer.frames)}/{self.recognizer.max_frames}",
            (padding + 100, padding + 20),
            0.35,
            (210, 225, 245),
            1,
        )

        if not self.recognizer.last_top:
            self._put_text(frame, "Show sign...", (padding + 8, padding + 55), 0.45, (0, 220, 255), 1)
            return frame

        bar_x = padding + int(160 * self.scale_factor)
        bar_w = max(int(60 * self.scale_factor), panel_w - int(170 * self.scale_factor))
        
        for idx, (label, score) in enumerate(self.recognizer.last_top[:3]):  # Show top 3
            y = padding + int(48 * self.scale_factor) + idx * line_height
            color = (70, 230, 90) if score >= self.confidence_threshold else (80, 170, 255)
            self._put_text(frame, f"{idx + 1}. {label}", (padding + 8, y), 0.45, color, 1 if idx > 0 else 2)
            self._put_text(frame, f"{score * 100:.0f}%", (padding + 120, y), 0.35, (235, 235, 235), 1)
            cv2.rectangle(frame, (bar_x, y - int(8 * self.scale_factor)), (bar_x + bar_w, y - int(2 * self.scale_factor)), (55, 60, 70), -1)
            cv2.rectangle(
                frame,
                (bar_x, y - int(8 * self.scale_factor)),
                (bar_x + int(bar_w * min(score, 1.0)), y - int(2 * self.scale_factor)),
                color,
                -1,
            )

        return frame

    @staticmethod
    def _put_text_unicode(frame, text, origin, font_size=20, color=(255, 255, 255)):
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            import os
            
            font_path = "arial.ttf"
            if os.name == 'nt':
                font_path = "LeelawUI.ttf" # Leelawadee UI supports Lao, Thai, Buginese, etc.
                
            # Convert BGR to RGB
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            try:
                font = ImageFont.truetype(font_path, int(font_size))
            except IOError:
                font = ImageFont.load_default()
                
            # PIL expects RGB, but color is BGR from OpenCV standard
            rgb_color = (color[2], color[1], color[0])
            draw.text(origin, text, font=font, fill=rgb_color)
            
            # Convert back to BGR
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            cv2.putText(frame, "Font Error", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return frame

    def _draw_integration_state(self, frame):
        h, w = frame.shape[:2]
        
        # Panel - smaller
        panel_h = int(85 * self.scale_factor)
        y0 = max(0, h - panel_h)
        self._blend_panel(frame, (0, y0), (w, h), alpha=0.82)

        # Do not use ascii_text for Vietnamese and Lao to preserve Unicode
        vietnamese = self.last_result["vietnamese"]
        lao = self.last_result["lao"]
        
        lines = [
            f"Token buffer: {self.pipeline.buffer.raw_sentence()}",
            f"Vietnamese: {vietnamese}",
            f"Status: {self.status_message}",
        ]
        if self.enable_translation and lao:
            lines.append(f"Lao: {lao}")
        lines.append("a accept | b build | c clear | SPACE reset vision | q quit")

        line_height = int(15 * self.scale_factor)
        for idx, line in enumerate(lines):
            y_pos = y0 + int(16 * self.scale_factor) + idx * line_height
            
            # Use unicode rendering for the translated strings
            if line.startswith("Vietnamese:") or line.startswith("Lao:"):
                frame = self._put_text_unicode(
                    frame, 
                    line[:118], 
                    (int(12 * self.scale_factor), y_pos - int(12 * self.scale_factor)), # Adjust Y because PIL draws from top-left, not bottom-left
                    font_size=int(15 * self.scale_factor), 
                    color=(255, 255, 255)
                )
            else:
                self._put_text(
                    frame,
                    line[:118],
                    (int(12 * self.scale_factor), y_pos),
                    0.35,
                    (255, 255, 255),
                    1,
                )
        return frame

    def run(self):
        recognizer = self.recognizer
        cap = recognizer._open_camera()
        if cap is None:
            print("ERROR: Could not open a working camera.")
            return

        print("Integrated realtime pipeline.")
        print("Keys: a=accept top, b=build, c=clear, SPACE=reset vision, q=quit.")

        # Get screen resolution
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide tkinter window
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
            self._update_scale_factor()
            print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        except:
            print(f"Using default resolution: {self.screen_width}x{self.screen_height}")

        # Set fullscreen window
        window_name = "Integrated Sign Language Pipeline"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            # Resize to full screen
            cv2.resizeWindow(window_name, self.screen_width, self.screen_height)
            # Move to top-left corner
            cv2.moveWindow(window_name, 0, 0)
            print(f"✅ Window set to fullscreen: {self.screen_width}x{self.screen_height}")
        else:
            print("Running in windowed mode")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = recognizer.holistic.process(rgb)
            rgb.flags.writeable = True

            if recognizer._hands_visible(results):
                recognizer.frames.append(recognizer._extract_frame_landmarks(results))
                recognizer.frame_counter += 1
                if recognizer.frame_counter % recognizer.predict_every == 0:
                    recognizer.last_top = recognizer.predict()
                    self._update_stability()

            frame = self._draw_landmarks(frame, results)
            frame = self._draw_prediction_panel(frame)
            frame = self._draw_integration_state(frame)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                self._accept_top_prediction()
            if key == ord("b"):
                if not self.is_building:
                    self.is_building = True
                    self.status_message = "Building sentence and translating..."
                    
                    def build_thread():
                        print("\n" + "=" * 70, flush=True)
                        print("🔄 BUILDING SENTENCE...", flush=True)
                        print("=" * 70, flush=True)
                        result = self.pipeline.build()
                        self.last_result = result
                        print("✅ Token buffer:", result["raw"], flush=True)
                        print("✅ Vietnamese:", result["vietnamese"], flush=True)
                        if self.enable_translation:
                            if result["lao"]:
                                print("✅ Lao:", result["lao"], flush=True)
                            else:
                                print("⚠️ Lao: (empty or failed)", flush=True)
                        print("=" * 70 + "\n", flush=True)
                        self.status_message = "Translation complete."
                        self.is_building = False
                        
                    threading.Thread(target=build_thread, daemon=True).start()
                else:
                    print("Already building, please wait...", flush=True)
            if key == ord("c"):
                self.pipeline.clear()
                self.last_result = {"raw": "", "vietnamese": "", "lao": ""}
                self.last_sign = None
                print("\nCleared text buffer.")
            if key == 32:
                recognizer.frames.clear()
                recognizer.last_top = []
                recognizer.frame_counter = 0
                self.prediction_history.clear()
                self.status_message = "Vision buffer reset."

        print("\nClosing...")
        cap.release()
        cv2.destroyAllWindows()
        recognizer.holistic.close()


def main():
    parser = argparse.ArgumentParser(description="Realtime ASL-250 + Vietnamese NLP + Vietnamese-Lao translation")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--no-translation", action="store_true")
    parser.add_argument("--use-bartpho", action="store_true")
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--stable-predictions", type=int, default=3)
    parser.add_argument("--windowed", action="store_true", help="Windowed mode (default is fullscreen)")
    args = parser.parse_args()

    app = IntegratedRealtimeRecognizer(
        camera=args.camera,
        confidence_threshold=args.confidence,
        enable_translation=not args.no_translation,
        use_bartpho=args.use_bartpho,
        auto_accept=args.auto_accept,
        stable_predictions=args.stable_predictions,
        fullscreen=not args.windowed,
    )
    app.run()


if __name__ == "__main__":
    main()

