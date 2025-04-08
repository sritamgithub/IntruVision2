import cv2
import threading
import queue
import time
from datetime import datetime
import numpy as np
import os
from collections import deque, defaultdict

class VideoStream:
    def __init__(self, src=0, name="WebcamVideoStream", resolution=(640, 480)):
        """Initialize the video stream"""
        self.stream = cv2.VideoCapture(src)
        
        # Set resolution for better performance
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Set buffersize to 1 for lower latency
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.name = name
        self.stopped = False
        self.frame = None
        self.frame_queue = queue.Queue(maxsize=2)  # Buffer only the most recent frames
        
        # Start the thread to read frames
        self.thread = threading.Thread(target=self.update, name=self.name)
        self.thread.daemon = True
    
    def start(self):
        """Start the thread to read frames"""
        self.thread.start()
        return self
    
    def update(self):
        """Read frames from the stream continuously"""
        while True:
            if self.stopped:
                break
            
            ret, frame = self.stream.read()
            
            if not ret:
                self.stopped = True
                break
            
            # Clear the queue and add the new frame
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
    
    def read(self):
        """Return the most recent frame"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the thread and release resources"""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()

class DualStreamSystem:
    """
    A dual-stream system that separates UI rendering from face recognition
    - UI Stream: Fast, lightweight, only shows basic face detection boxes
    - Recognition Stream: Slower, does full face recognition and DB operations
    """
    def __init__(self, face_recognizer, db, alert_system, src=0, 
                 detection_interval=5, frames_per_detection=10):
        """Initialize the dual-stream system"""
        self.face_recognizer = face_recognizer
        self.db = db
        self.alert_system = alert_system
        self.src = src
        self.detection_interval = detection_interval  # seconds - how often to run recognition
        self.frames_per_detection = frames_per_detection
        
        # Camera streams - separate instances for UI and recognition
        self.ui_stream = None  # Fast stream for UI
        self.recog_stream = None  # Slower stream for recognition
        
        # Threading
        self.stopped = False
        self.ui_thread = None
        self.recog_thread = None
        
        # Frames and results
        self.ui_frame = None  # Frame with basic boxes for UI display
        self.recognition_results = {
            'names': [],
            'locations': [],
            'result_time': 0,
            'status': 'idle'
        }
        
        # Get the path to OpenCV's data directory
        haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)

        # Face detection cascade for UI stream
        # self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Buffer for recognition stream
        self.frame_buffer = deque(maxlen=frames_per_detection)
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Detection tracking & log prevention
        self.last_recognition_time = 0
        self.person_log_cache = defaultdict(float)
    
    def start(self):
        """Start both streams and processing threads"""
        # Start UI camera stream (higher FPS, minimal processing)
        self.ui_stream = VideoStream(src=self.src, resolution=(640, 480)).start()
        
        # Start recognition camera stream (lower FPS, more intensive processing)
        self.recog_stream = VideoStream(src=self.src, resolution=(640, 480)).start()
        
        # Allow cameras to warm up
        time.sleep(1.0)
        
        # Start UI thread
        self.ui_thread = threading.Thread(target=self.process_ui_stream)
        self.ui_thread.daemon = True
        self.ui_thread.start()
        
        # Start recognition thread
        self.recog_thread = threading.Thread(target=self.process_recognition_stream)
        self.recog_thread.daemon = True
        self.recog_thread.start()
        
        return self
    
    def process_ui_stream(self):
        """Process frames for UI display - fast, simple detection"""
        while not self.stopped:
            # Read frame
            frame = self.ui_stream.read()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Create a copy for processing
            display_frame = frame.copy()
            
            # Check if we need to incorporate recognition results
            with self.lock:
                recognition_active = (time.time() - self.recognition_results['result_time']) < 2
                recognition_names = self.recognition_results['names']
                recognition_locations = self.recognition_results['locations']
            
            if recognition_active and recognition_names and recognition_locations:
                # Use recognition results for display if fresh enough
                for i, (top, right, bottom, left) in enumerate(recognition_locations):
                    if i < len(recognition_names):
                        name = recognition_names[i]
                        
                        # Scale back coordinates (they were halved in recognition)
                        top *= 2
                        right *= 2
                        bottom *= 2
                        left *= 2
                        
                        # Get color based on recognition type
                        if name == "Unknown":
                            color = (0, 0, 255)  # Red for unknown
                            label = "Unknown"
                        else:
                            # Check if person is a guest
                            cursor = self.db.get_connection().cursor()
                            cursor.execute("""
                                SELECT g.id FROM GUEST g
                                JOIN PERSON p ON g.person_id = p.id
                                WHERE p.name = ? AND date('now') BETWEEN g.start_date AND g.end_date
                            """, (name,))
                            guest = cursor.fetchone()
                            
                            if guest:
                                color = (0, 255, 255)  # Yellow for guest
                                label = f"Guest: {name}"
                            else:
                                color = (0, 255, 0)  # Green for known
                                label = name
                        
                        # Draw face rectangle and label
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(display_frame, label, (left + 6, bottom - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Fast face detection for real-time UI feedback (no recognition)
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                # Draw empty boxes for faces
                for (x, y, w, h) in faces:
                    # Default yellow boxes for unrecognized faces
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Add processing status indicator
            status_text = f"Recognition: {self.recognition_results['status']}"
            cv2.putText(display_frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Update the display frame
            self.ui_frame = display_frame
            
            # Control frame rate for smoother UI
            time.sleep(0.03)  # ~30fps
    
    def process_recognition_stream(self):
        """Process frames for face recognition - slower, more intensive processing"""
        while not self.stopped:
            # Read frame
            frame = self.recog_stream.read()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Add to buffer for batch processing
            self.frame_buffer.append(frame.copy())
            
            # Check if it's time for recognition
            current_time = time.time()
            if (current_time - self.last_recognition_time) >= self.detection_interval:
                # Begin recognition
                with self.lock:
                    self.recognition_results['status'] = 'processing'
                
                # Process batch of frames
                if len(self.frame_buffer) >= max(1, self.frames_per_detection // 2):
                    # Update status
                    with self.lock:
                        self.recognition_results['status'] = 'analyzing'
                    
                    # Get frames to process
                    frames_to_process = list(self.frame_buffer)
                    
                    try:
                        # Process frames in batch
                        consensus_names, face_locations, name_counts = self.face_recognizer.process_frames_batch(frames_to_process)
                        
                        # Handle detection results (DB logging, alerts, etc.)
                        self._handle_detection_results(consensus_names, face_locations, frames_to_process[-1] if frames_to_process else None)
                        
                        # Update recognition results for UI thread
                        with self.lock:
                            self.recognition_results = {
                                'names': consensus_names,
                                'locations': face_locations,
                                'result_time': time.time(),
                                'status': 'completed'
                            }
                    except Exception as e:
                        print(f"Error in recognition processing: {e}")
                        with self.lock:
                            self.recognition_results['status'] = 'error'
                    
                    # Clear buffer
                    self.frame_buffer.clear()
                
                # Update last recognition time
                self.last_recognition_time = current_time
            
            # Less aggressive sleep to reduce CPU usage
            time.sleep(0.1)
    
    def _handle_detection_results(self, names, face_locations, frame):
        """Process recognition results for logging and alerts"""
        current_time = time.time()
        
        for i, name in enumerate(names):
            # Skip if this person was recently logged (except for Unknown)
            if name != "Unknown":
                last_log_time = self.person_log_cache.get(name, 0)
                if current_time - last_log_time < 60:  # 1-minute window
                    continue  # Skip logging
                
                # Update log cache
                self.person_log_cache[name] = current_time
            
            # Log each detection
            if name != "Unknown":
                # Get person ID
                person = self.db.get_person_by_name(name)
                if person:
                    person_id = person['id']
                    # Check if it's a guest
                    cursor = self.db.get_connection().cursor()
                    cursor.execute("""
                        SELECT id FROM GUEST 
                        WHERE person_id = ? AND date('now') BETWEEN start_date AND end_date
                    """, (person_id,))
                    guest = cursor.fetchone()
                    
                    status = 'guest' if guest else 'recognized'
                    self.alert_system.handle_detection(person_id, name, status, None)
            else:
                # Handle unknown face (always log these)
                if i < len(face_locations) and frame is not None:
                    # Save unknown face image
                    face_img_path = self.face_recognizer.save_unknown_face(frame, face_locations[i])
                    # Send alert
                    self.alert_system.handle_detection(None, "Unknown", "unknown", face_img_path)
        
        # Clean up expired cache entries
        expired_names = [n for n, t in self.person_log_cache.items() 
                        if current_time - t > 120]  # 2-minute expiry
        for name in expired_names:
            del self.person_log_cache[name]
    
    def read(self):
        """Return the current frame for UI display"""
        return self.ui_frame if self.ui_frame is not None else None
    
    def get_detected_names(self):
        """Return the most recent detected names"""
        with self.lock:
            return self.recognition_results['names']
    
    def stop(self):
        """Stop all streams and threads"""
        self.stopped = True
        
        # Stop UI stream
        if self.ui_stream:
            self.ui_stream.stop()
        
        # Stop recognition stream
        if self.recog_stream:
            self.recog_stream.stop()
        
        # Join threads with timeout
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1.0)
        
        if self.recog_thread and self.recog_thread.is_alive():
            self.recog_thread.join(timeout=1.0)
                 
class FaceDetectionStream:
    def __init__(self, face_recognizer, db, alert_system, src=0, detection_interval=2000, frames_per_detection=10):
        """Initialize the face detection stream"""
        self.face_recognizer = face_recognizer
        self.db = db
        self.alert_system = alert_system
        self.src = src
        self.detection_interval = detection_interval  # milliseconds
        self.frames_per_detection = frames_per_detection  # Number of frames to analyze for consensus
        self.stream = None
        self.processing_thread = None
        self.stopped = False
        self.frame = None
        self.processed_frame = None
        self.last_detection_time = 0
        self.detected_names = []
        self.face_locations = []
        
        # Buffer to store frames for batch processing
        self.frame_buffer = deque(maxlen=frames_per_detection)
    
    def start(self):
        """Start the video stream and processing thread"""
        self.stream = VideoStream(src=self.src, resolution=(640, 480)).start()
        time.sleep(1.0)  # Allow camera to warm up
        
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return self
    
    def process_frames(self):
        """Process frames for face detection and recognition"""
        while not self.stopped:
            # Read the next frame
            frame = self.stream.read()
            
            if frame is None:
                time.sleep(0.01)  # Avoid tight loop if no frame
                continue
            
            # Make a copy of the frame for processing
            current_frame = frame.copy()
            
            # Add frame to buffer
            self.frame_buffer.append(current_frame)
            
            # Check if it's time for a new detection
            current_time = int(time.time() * 1000)
            if (current_time - self.last_detection_time) >= self.detection_interval and len(self.frame_buffer) > 0:
                # Process a batch of frames
                if len(self.frame_buffer) >= self.frames_per_detection / 2:  # At least half the required frames
                    # Copy frames from buffer for processing
                    frames_to_process = list(self.frame_buffer)
                    
                    # Process frames in batch for better consensus
                    consensus_names, face_locations, name_counts = self.face_recognizer.process_frames_batch(frames_to_process)
                    
                    # Handle each detected face
                    for i, name in enumerate(consensus_names):
                        # Process recognition result
                        self._handle_detection(name, current_frame, face_locations[i] if i < len(face_locations) else None)
                    
                    # Store detection results
                    self.detected_names = consensus_names
                    self.face_locations = face_locations
                    
                    # Update detection time
                    self.last_detection_time = current_time
                    
                    # Clear buffer
                    self.frame_buffer.clear()
                else:
                    # Not enough frames yet, process single frame
                    processed_frame, names, locations = self.face_recognizer.recognize_faces(current_frame)
                    
                    # Store detection results
                    self.detected_names = names
                    self.face_locations = locations
                    
                    # Update processed frame
                    self.processed_frame = processed_frame
                    
                    # Update detection time
                    self.last_detection_time = current_time
            
            # If we have a processed frame, use it
            if self.processed_frame is not None:
                self.frame = self.processed_frame
            else:
                # Draw any previous detection results on the current frame
                if self.face_locations and self.detected_names:
                    for i, (top, right, bottom, left) in enumerate(self.face_locations):
                        if i < len(self.detected_names):
                            name = self.detected_names[i]
                            
                            # Scale back coordinates since we resized the frame
                            top *= 2
                            right *= 2
                            bottom *= 2
                            left *= 2
                            
                            # Check if person is a guest
                            cursor = self.db.get_connection().cursor()
                            cursor.execute("""
                                SELECT g.id FROM GUEST g
                                JOIN PERSON p ON g.person_id = p.id
                                WHERE p.name = ? AND date('now') BETWEEN g.start_date AND g.end_date
                            """, (name,))
                            guest = cursor.fetchone()
                            
                            # Set color and label based on recognition status
                            if name == "Unknown":
                                color = (0, 0, 255)  # Red for unknown
                                label = "Unknown"
                            elif guest:
                                color = (0, 255, 255)  # Yellow for guest
                                label = f"Guest: {name}"
                            else:
                                color = (0, 255, 0)  # Green for known
                                label = name
                            
                            # Draw rectangle around face
                            cv2.rectangle(current_frame, (left, top), (right, bottom), color, 2)
                            
                            # Draw label background
                            cv2.rectangle(current_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            cv2.putText(current_frame, label, (left + 6, bottom - 6), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                self.frame = current_frame
    
    def _handle_detection(self, name, frame, face_location):
        """Handle a face detection event"""
        if name != "Unknown":
            # Get person ID
            person = self.db.get_person_by_name(name)
            if person:
                person_id = person['id']
                # Check if it's a guest
                cursor = self.db.get_connection().cursor()
                cursor.execute("""
                    SELECT id FROM GUEST 
                    WHERE person_id = ? AND date('now') BETWEEN start_date AND end_date
                """, (person_id,))
                guest = cursor.fetchone()
                
                status = 'guest' if guest else 'recognized'
                self.alert_system.handle_detection(person_id, name, status, None)
        else:
            # Handle unknown face
            if face_location:
                # Save unknown face image
                face_img_path = self.face_recognizer.save_unknown_face(frame, face_location)
                # Send alert
                self.alert_system.handle_detection(None, "Unknown", "unknown", face_img_path)
    
    def read(self):
        """Return the most recent processed frame"""
        return self.frame if self.frame is not None else None
    
    def get_detected_names(self):
        """Return the most recent detected names"""
        return self.detected_names
    
    def stop(self):
        """Stop the stream and release resources"""
        self.stopped = True
        if self.stream:
            self.stream.stop()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)