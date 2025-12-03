import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

class WeedDetector:
    def __init__(self, model_path="C:\\Users\\Chivukula\\Projects\\weed\\weed-detector\\results\\train\\weights\\best.pt"):
        """Initialize the weed detection system"""
        print("🌿 Loading Weed Detection Model...")
        
        # Load the trained model
        self.model = YOLO(model_path)
        self.model.conf = 0.5  # Confidence threshold
        self.model.iou = 0.45  # IoU threshold
        
        # Class names - we only care about weeds (class 0)
        self.class_names = ['weed', 'plant']
        self.weed_color = (0, 255, 0)  # Green for weeds
        
        print("✅ Model loaded successfully!")
        print(f"   Weed class: {self.class_names[0]} (Class 0)")
        print(f"   Confidence threshold: {self.model.conf}")
        
    def setup_webcam(self, camera_id=0): #0 for internal webcam, 1 for external USB
        """Setup webcam capture"""
        self.cap = cv2.VideoCapture(camera_id)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            return False
        
        print("📷 Webcam initialized successfully")
        return True
    
    def draw_weed_detections(self, frame, results):
        """Draw bounding boxes ONLY for weeds and return coordinates (includes center)"""
        detections = results[0].boxes
        
        weed_coordinates = []
        weed_count = 0
        
        if detections is not None and len(detections) > 0:
            for box in detections:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Only process weeds (class 0)
                if cls == 0:  # weed class
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Compute center point
                    center_x = x1 + width // 2
                    center_y = y1 + height // 2
                    
                    # Store coordinates: [x, y, width, height, confidence, center_x, center_y]
                    weed_coordinates.append([x1, y1, width, height, conf, center_x, center_y])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.weed_color, 3)
                    
                    # Draw label
                    label = f"Weed: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), self.weed_color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw center point
                    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 0), -1)
                    
                    weed_count += 1
        
        return frame, weed_coordinates, weed_count
    
    def print_weed_coordinates(self, weed_coordinates, frame_number):
        """Print weed coordinates including center point"""
        if weed_coordinates:
            print(f"\n📍 Frame {frame_number}: Detected {len(weed_coordinates)} weeds")
            for i, coord in enumerate(weed_coordinates):
                x, y, w, h, conf, cx, cy = coord
                print(f"   - Weed {i+1}: [x: {x}, y: {y}, w: {w}, h: {h}] center: ({cx}, {cy}) conf: {conf:.3f}")
        else:
            print(f"📍 Frame {frame_number}: No weeds detected")
    
    def run_detection(self):
        """Run real-time weed-only detection"""
        if not self.setup_webcam():
            return
        
        print("\n🎮 Starting real-time WEED-ONLY detection...")
        print("   Press 'q' to quit")
        print("   Press 's' to save current frame")
        print("   Press 'c' to change confidence threshold")
        print("   Press 'd' to toggle coordinate display")
        print("   Press 'v' to start/stop video recording")
        print("-" * 50)
        
        # Ensure results directory exists
        results_dir = Path("weed-detector/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # FPS calculation
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        frame_number = 0
        show_coordinates = True

        # Video recording state
        recording = False
        video_writer = None
        video_filename = None
        video_size = None
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_number += 1
            
            # Run detection
            start_time = time.time()
            results = self.model(frame, verbose=False)
            inference_time = time.time() - start_time
            
            # Draw ONLY weed detections and get coordinates
            frame, weed_coordinates, weed_count = self.draw_weed_detections(frame, results)
            
            # Print coordinates to console
            if show_coordinates:
                self.print_weed_coordinates(weed_coordinates, frame_number)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # If recording, write the frame (ensure writer initialized)
            if recording and video_writer is not None:
                try:
                    h_frame, w_frame = frame.shape[:2]
                    # video_size stored as (width, height)
                    if video_size is not None and (w_frame, h_frame) != video_size:
                        frame_to_write = cv2.resize(frame, video_size)
                    else:
                        frame_to_write = frame
                    video_writer.write(frame_to_write)
                except Exception as e:
                    print(f"❌ Error writing frame to video: {e}")

            # Display information
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Weeds: {weed_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {self.model.conf:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Coord: {'ON' if show_coordinates else 'OFF'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            if recording:
                cv2.putText(frame, f"REC ●", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Weed Detection - WEEDS ONLY', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = results_dir / f"weed_capture_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"💾 Frame saved: {filename}")
            elif key == ord('c'):
                # Change confidence threshold
                try:
                    new_conf = float(input("Enter new confidence threshold (0.1-0.9): "))
                    if 0.1 <= new_conf <= 0.9:
                        self.model.conf = new_conf
                        print(f"✅ Confidence threshold updated to: {new_conf}")
                    else:
                        print("❌ Please enter a value between 0.1 and 0.9")
                except ValueError:
                    print("❌ Invalid input")
            elif key == ord('d'):
                # Toggle coordinate display
                show_coordinates = not show_coordinates
                status = "ON" if show_coordinates else "OFF"
                print(f"✅ Coordinate display: {status}")
            elif key == ord('v'):
                # Toggle video recording
                if not recording:
                    # Start recording
                    timestamp = int(time.time())
                    out_fps = int(round(fps)) if fps > 0 else 20
                    h_frame, w_frame = frame.shape[:2]
                    video_size = (w_frame, h_frame)  # (width, height)

                    # Try MP4 then AVI (XVID)
                    attempts = [
                        (results_dir / f"weed_recording_{timestamp}.mp4", 'mp4v'),
                        (results_dir / f"weed_recording_{timestamp}.avi", 'XVID'),
                    ]
                    opened = False
                    for candidate, codec in attempts:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            writer = cv2.VideoWriter(str(candidate), fourcc, out_fps, video_size)
                            if writer.isOpened():
                                video_writer = writer
                                video_filename = candidate
                                opened = True
                                break
                            else:
                                writer.release()
                        except Exception:
                            try:
                                writer.release()
                            except Exception:
                                pass

                    if not opened:
                        video_writer = None
                        video_filename = None
                        video_size = None
                        print("❌ Error: Could not open any video writer (tried mp4v and XVID).")
                    else:
                        recording = True
                        print(f"🔴 Started recording: {video_filename} (fps={out_fps}, size={video_size})")
                else:
                    # Stop recording
                    recording = False
                    if video_writer is not None:
                        try:
                            video_writer.release()
                        except Exception as e:
                            print(f"❌ Error releasing video writer: {e}")
                        # confirm file
                        if video_filename is not None and video_filename.exists():
                            print(f"⏹️  Stopped recording, saved: {video_filename}")
                        else:
                            print(f"⏹️  Stopped recording, but file not found: {video_filename}")
                        video_writer = None
                        video_filename = None
                        video_size = None
        
        # Cleanup
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Weed detection stopped")

def test_weed_detection_with_images():
    """Test weed detection with sample images and output coordinates"""
    print("\n🧪 Testing WEED-ONLY detection with sample images...")
    
    detector = WeedDetector()
    
    test_images_dir = Path("weed-detector/data/processed/images/test")
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if image_files:
            for i, image_path in enumerate(image_files[:3]):
                print(f"\n📷 Testing with: {image_path.name}")
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"❌ Could not load image: {image_path}")
                    continue
                
                # Run detection
                results = detector.model(image)
                
                # Get weed coordinates
                image_with_boxes, weed_coordinates, weed_count = detector.draw_weed_detections(image, results)
                
                # Print coordinates
                detector.print_weed_coordinates(weed_coordinates, i+1)
                
                # Show image
                cv2.imshow(f"Weed Test - {image_path.name}", image_with_boxes)
                cv2.waitKey(3000)  # Show for 3 seconds
                cv2.destroyAllWindows()
        else:
            print("❌ No test images found")
    else:
        print("❌ Test images directory not found")

if __name__ == "__main__":
    print("🌿 AI Weed Detection System - WEEDS ONLY")
    print("=" * 50)
    
    # Test with sample images first
    test_weed_detection_with_images()
    
    # Start real-time detection
    detector = WeedDetector()
    detector.run_detection()