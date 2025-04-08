import face_recognition
import cv2
import pickle
import numpy as np
import os
from imutils import paths
from datetime import datetime
import sqlite3
import shutil
import time
from collections import Counter


class FaceRecognitionSystem:
    def __init__(self, encodings_path="model/encodings.pickle", 
                 cascade_path="haarcascade_frontalface_default.xml",
                 detection_method="hog", tolerance=0.5):
        """Initialize the face recognition system"""
        self.encodings_path = encodings_path
        self.cascade_path = cascade_path
        self.detection_method = detection_method
        self.tolerance = tolerance
        self.data = None
        self.detector = None
        self.db_connection = sqlite3.connect('students.db', check_same_thread=False)
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
        
        # Load the face detection model
        self.load_detector()
        
        # Load the encodings
        self.load_encodings()
    
    def load_detector(self):
        """Load the face detector cascade"""
        haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.detector = cv2.CascadeClassifier(haar_cascade_path)
   
    def load_encodings(self):
        """Load the face encodings from pickle file"""
        try:
            self.data = pickle.loads(open(self.encodings_path, "rb").read())
            print(f"Loaded {len(self.data['encodings'])} face encodings")
        except Exception as e:
            print(f"Error loading encodings: {e}")
            # Initialize empty data structure if file doesn't exist
            self.data = {"encodings": [], "names": []}
    
    def generate_encodings(self, images_dir="static/images/users"):
        """Generate encodings from images in the specified directory"""
        print("[INFO] Generating face encodings...")
        image_paths = list(paths.list_images(images_dir))
        
        known_encodings = []
        known_names = []
        encoding_count = 0
        
        # Create a dictionary to track encodings per person
        person_encoding_count = {}
        
        for (i, image_path) in enumerate(image_paths):
            print(f"[INFO] Processing image {i+1}/{len(image_paths)} - {image_path}")
            
            try:
                # Extract person name from directory name
                name = os.path.basename(os.path.dirname(image_path))
                
                # Track encodings per person
                if name not in person_encoding_count:
                    person_encoding_count[name] = 0
                
                # Load and convert image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image: {image_path}")
                    continue
                    
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                boxes = face_recognition.face_locations(rgb, model=self.detection_method)
                
                if not boxes:
                    print(f"No faces detected in {image_path}")
                    continue
                
                # Generate encodings
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                # Add encodings and names
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)
                    person_encoding_count[name] += 1
                    encoding_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Save encodings to file
        print("[INFO] Serializing encodings...")
        self.data = {"encodings": known_encodings, "names": known_names}
        
        # Save to file
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(self.data))
        
        print(f"[INFO] Encodings generated for {len(person_encoding_count)} persons:")
        for person, count in person_encoding_count.items():
            print(f"  - {person}: {count} encodings")
            
        return encoding_count
    

    def align_face(self, image, face_landmarks):
        """Align face based on eyes for better recognition accuracy"""
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")
        
        # Calculate angle
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get center of face
        center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Rotate to align eyes horizontally
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
        
        return aligned


    def recognize_faces(self, frame):
        """
        Recognize faces in the given frame with improved accuracy
        Returns: (processed_frame, detected_names, face_locations)
        """
        # Make a copy of the frame
        frame_copy = frame.copy()
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5)
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_method)
        
        # If no faces found, return original frame
        if not face_locations:
            return frame_copy, [], []
        
        # Generate encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        names = []
        confidence_scores = []
        
        # Process each face
        for (i, encoding) in enumerate(face_encodings):
            # Compare with known encodings
            matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=self.tolerance)
            name = "Unknown"
            confidence = 0.0
            
            # Find the best match
            if True in matches and len(self.data["encodings"]) > 0:
                # Get indexes of all matches
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                
                # Calculate face distances for matched faces
                face_distances = face_recognition.face_distance(
                    [self.data["encodings"][idx] for idx in matched_idxs], 
                    encoding
                )
                
                # Get the index of the closest match
                if len(face_distances) > 0:
                    best_match_idx = matched_idxs[np.argmin(face_distances)]
                    name = self.data["names"][best_match_idx]
                    confidence = 1.0 - min(face_distances)
            
            # FIX: This is the critical bug - it was appending 'Unknown' for all valid names
            # Only add name if it's valid and has sufficient confidence
            if name != "Unknown" and confidence >= 0.55:  # Increased confidence threshold
                names.append(name)
            else:
                names.append("Unknown")
                
            confidence_scores.append(confidence)
            
            # Draw results on frame
            # Scale back coordinates since we resized the frame
            (top, right, bottom, left) = face_locations[i]
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Check if person is a guest
            cursor = self.db_connection.cursor()
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
                
            # Add confidence if it's a known person
            if name != "Unknown" and confidence > 0:
                confidence_percentage = int(confidence * 100)
                label = f"{label} ({confidence_percentage}%)"
            
            # Draw rectangle around face
            cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame_copy, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame_copy, label, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame_copy, names, face_locations



    def process_frames_batch(self, frames, min_confidence=0.6):
        """
        Process multiple frames to get consensus-based recognition
        This improves accuracy by analyzing multiple frames and taking the most common result
        
        Args:
            frames (list): List of frames to process
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            tuple: (consensus_names, face_locations, name_counts)
        """
        from collections import Counter
        
        all_detections = []
        
        # Process each frame
        for frame in frames:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_method)
            
            # Skip if no faces found
            if not face_locations:
                continue
                
            # Generate encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Process each face
            for i, encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=self.tolerance)
                name = "Unknown"
                confidence = 0.0
                
                if True in matches and len(self.data["encodings"]) > 0:
                    # Get matched indexes
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        [self.data["encodings"][idx] for idx in matched_idxs], 
                        encoding
                    )
                    
                    # Get best match
                    if len(face_distances) > 0:
                        best_match_idx = matched_idxs[np.argmin(face_distances)]
                        name = self.data["names"][best_match_idx]
                        confidence = 1.0 - min(face_distances)
                
                # Only consider detections with sufficient confidence
                if name != "Unknown" and confidence < min_confidence:
                    name = "Unknown"
                    
                # Store detection with location
                all_detections.append({
                    'name': name,
                    'confidence': confidence,
                    'location': face_locations[i],
                    'encoding': encoding
                })
        
        # If we have no detections, return empty results
        if not all_detections:
            return [], [], []
        
        # Group detections by similarity (face position)
        grouped_detections = self._group_detections_by_position(all_detections)
        
        # Get consensus for each face
        consensus_results = []
        for group in grouped_detections:
            # Count occurrences of each name in this group
            name_counter = Counter([d['name'] for d in group])
            
            # Get the most common name
            most_common = name_counter.most_common(1)
            if most_common:
                consensus_name, count = most_common[0]
                
                # Calculate average confidence for this name
                confidences = [d['confidence'] for d in group if d['name'] == consensus_name]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Only consider it a match if it appears in multiple frames AND has sufficient confidence
                # This is a key improvement for accuracy
                if count >= 3 and avg_confidence >= min_confidence and consensus_name != "Unknown":
                    final_name = consensus_name
                else:
                    final_name = "Unknown"
                    
                # Get the last detected location for this face
                last_location = group[-1]['location']
                
                consensus_results.append({
                    'name': final_name,
                    'confidence': avg_confidence,
                    'location': last_location,
                    'count': count,
                    'total': len(group)
                })
        
        # Extract results
        consensus_names = [r['name'] for r in consensus_results]
        face_locations = [r['location'] for r in consensus_results]
        name_counts = [(r['name'], r['count'], r['total']) for r in consensus_results]
        
        return consensus_names, face_locations, name_counts
    def _group_detections_by_position(self, detections, overlap_threshold=0.5):
        """
        Group detections that appear to be the same face in different frames
        based on position overlap
        
        Args:
            detections (list): List of detection dictionaries
            overlap_threshold (float): Minimum IOU (intersection over union) to consider as same face
            
        Returns:
            list: List of grouped detections
        """
        if not detections:
            return []
        
        # Sort by confidence (higher confidence first)
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        # Initialize groups with the first detection
        groups = [[detections[0]]]
        
        # For each remaining detection
        for detection in detections[1:]:
            # Try to find a matching group
            matched = False
            for group in groups:
                # Check if this detection matches this group
                for group_detection in group:
                    if self._boxes_overlap(detection['location'], group_detection['location'], overlap_threshold):
                        # Add to this group
                        group.append(detection)
                        matched = True
                        break
                if matched:
                    break
                    
            # If no match found, create a new group
            if not matched:
                groups.append([detection])
        
        return groups


    def _boxes_overlap(self, box1, box2, threshold=0.5):
        """
        Calculate if two face bounding boxes overlap significantly
        
        Args:
            box1, box2: Bounding boxes in format (top, right, bottom, left)
            threshold: Minimum overlap ratio to consider as match
            
        Returns:
            bool: True if boxes overlap significantly
        """
        # Convert format from (top, right, bottom, left) to (x1, y1, x2, y2)
        box1_top, box1_right, box1_bottom, box1_left = box1
        box2_top, box2_right, box2_bottom, box2_left = box2
        
        # Calculate area of each box
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        
        # Calculate intersection coordinates
        intersect_left = max(box1_left, box2_left)
        intersect_top = max(box1_top, box2_top)
        intersect_right = min(box1_right, box2_right)
        intersect_bottom = min(box1_bottom, box2_bottom)
        
        # Check if intersection exists
        if intersect_right < intersect_left or intersect_bottom < intersect_top:
            return False
        
        # Calculate intersection area
        intersection_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU (Intersection over Union)
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou > threshold

    def save_unknown_face(self, frame, face_location):
        """Save unknown face for later processing"""
        top, right, bottom, left = face_location
        # Scale back coordinates since we resized the frame
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        
        # Make sure coordinates are within frame boundaries
        height, width = frame.shape[:2]
        top = max(0, top)
        left = max(0, left)
        bottom = min(height, bottom)
        right = min(width, right)
        
        # Extract face image
        face_img = frame[top:bottom, left:right]
        
        # Create directory if not exists
        os.makedirs("static/images/unknown", exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"static/images/unknown/unknown_{timestamp}.jpg"
        cv2.imwrite(file_path, face_img)
        
        return file_path
    
    def add_person(self, name, images):
        """Add a new person with multiple face images"""
        # Create directory for person
        person_dir = f"static/images/users/{name}"
        os.makedirs(person_dir, exist_ok=True)
        
        # Save images
        saved_paths = []
        for i, img in enumerate(images):
            file_path = f"{person_dir}/{name}_{i+1}.jpg"
            if hasattr(img, 'save'):  # If it's a PIL Image
                img.save(file_path)
            else:  # If it's a numpy array/OpenCV image
                cv2.imwrite(file_path, img)
            saved_paths.append(file_path)
        
        # Generate new encodings
        self.generate_encodings()
        
        return saved_paths
    
    def delete_person(self, name):
        """Delete a person and their face data"""
        person_dir = f"static/images/users/{name}"
        
        # Delete directory if exists
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        
        # Update encodings
        self.generate_encodings()
        
        return True
        
    def get_training_stats(self):
        """Get statistics about the current training data"""
        if not self.data or not self.data['names']:
            return {
                'total_encodings': 0,
                'total_people': 0,
                'people': []
            }
            
        # Count encodings per person
        person_counts = Counter(self.data['names'])
        
        stats = {
            'total_encodings': len(self.data['encodings']),
            'total_people': len(person_counts),
            'people': [{'name': name, 'encodings': count} for name, count in person_counts.items()]
        }
        
        return stats