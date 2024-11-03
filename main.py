import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class FootballDrillAnalyzer:
    def __init__(self):
        # Load YOLO model for ball detection
        self.model = YOLO('yolov8n.pt')
        
        # Initialize parameters
        self.ball_positions = deque(maxlen=30)
        self.hit_count = 0
        self.intensity_window = deque(maxlen=50)
        self.MIN_HIT_VELOCITY = 5  # Minimum velocity to consider a hit
        self.COLORS = {
            'high': (0, 0, 255),    # Red
            'medium': (0, 165, 255), # Orange
            'low': (0, 255, 0),     # Green
            'text': (255, 255, 255)  # White
        }

    def detect_ball(self, frame):
        """Detect the football in the frame using YOLO."""
        results = self.model(frame, classes=[32])  # Class ID for sports ball
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            confidence = float(box.conf[0])
            if confidence > 0.5:  # Minimum confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return (center_x, center_y)
        return None

    def update_hit_count(self, current_position):
        """Update hit count and direction based on ball position changes."""
        if len(self.ball_positions) < 2:
            return None, None
        
        previous_position = self.ball_positions[-2]
        velocity = np.linalg.norm(np.array(current_position) - np.array(previous_position))
        
        if velocity >= self.MIN_HIT_VELOCITY:
            self.hit_count += 1
            self.intensity_window.append(velocity)  # Track hit intensity based on velocity
            
            # Calculate direction
            direction = "Unknown"
            dx = current_position[0] - previous_position[0]
            dy = current_position[1] - previous_position[1]
            if abs(dx) > abs(dy):
                direction = "Right" if dx > 0 else "Left"
            else:
                direction = "Down" if dy > 0 else "Up"
                
            return direction, velocity
        return None, None

    def process_frame(self, frame):
        """Process a single frame and return analyzed frame with overlays."""
        display_frame = frame.copy()
        
        ball_pos = self.detect_ball(frame)
        if ball_pos:
            self.ball_positions.append(ball_pos)
            direction, intensity = self.update_hit_count(ball_pos)
            # Draw ball position
            cv2.circle(display_frame, ball_pos, 10, (0, 255, 0), -1)
            
            # Display hit direction and intensity
            if direction:
                cv2.putText(display_frame, f"Direction: {direction}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            if intensity is not None:
                cv2.putText(display_frame, f"Hit Intensity: {intensity:.2f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS[self.get_current_intensity(intensity)], 2)

        # Draw metrics
        self.draw_metrics(display_frame)

        return display_frame

    def draw_metrics(self, frame):
        """Draw metrics on the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 1 - 0.7, 0)

        # Hit Count Display
        cv2.putText(frame, f"Hit Count: {self.hit_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def get_current_intensity(self, current_velocity):
        """Determine intensity level based on velocity."""
        if current_velocity >= 20:  # Threshold for high intensity
            return 'high'
        elif current_velocity >= 10:  # Threshold for medium intensity
            return 'medium'
        else:
            return 'low'

    def analyze_video(self, video_path, output_path):
        """Process entire video and save result."""
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)
            cv2.imshow('Football Drill Analysis', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    analyzer = FootballDrillAnalyzer()
    video_path = "Squares.mp4"  # Change this to your input video file name
    output_path = "sqaure.mp4"
    
    print("Starting analysis...")
    analyzer.analyze_video(video_path, output_path)
    print("Analysis completed. Check the output video.")

if __name__ == "__main__":
    main()
