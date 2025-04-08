from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify, session
import cv2
import os
from datetime import date
import time
import numpy as np
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import queue
import threading
import base64
from io import BytesIO
from PIL import Image
import json
from flask import has_request_context
# Import modules
from modules.database import Database
from modules.face_recognition import FaceRecognitionSystem
from modules.alert_system import AlertSystem
from modules.stream import DualStreamSystem, VideoStream
from modules.auth import init_auth_db, verify_credentials, login_required, change_password
from flask import Flask, send_from_directory

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'face_recognition_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MIN_FACE_IMAGES'] = 5  # Minimum number of face images required
app.config['DETECTION_INTERVAL'] = 5  # Detection interval in seconds
app.config['STATIC_FOLDER'] = 'static'
app.config['FRAMES_PER_DETECTION'] = 10  # Number of frames to analyze for each detection

# Add current date to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Initialize database
db = Database()

# Initialize authentication
init_auth_db(db.get_connection())

# Initialize face recognition system
face_recognition_system = FaceRecognitionSystem()

# Initialize alert system
alert_system = AlertSystem(db)

# Initialize face detection stream
video_stream = None

# Global variables
is_detecting = True
is_training = False
training_queue = queue.Queue()
training_thread = None
recent_detections_cache = {}
admin_camera = None

# Start background worker for face model training
def background_training_worker():
    global is_training
    while True:
        try:
            # Wait for training task
            training_task = training_queue.get()
            if training_task == "EXIT":
                break
                
            is_training = True
            print("Starting face recognition model training...")
            
            # Perform model training
            face_recognition_system.generate_encodings()
            
            is_training = False
            print("Face recognition model training completed")
            
            # Mark the task as done
            training_queue.task_done()
        except Exception as e:
            print(f"Error in training worker: {e}")
            is_training = False

# Start the training thread
training_thread = threading.Thread(target=background_training_worker)
training_thread.daemon = True
training_thread.start()

@app.route('/static/models/face-api/<path:filename>')
def serve_model(filename):
    return send_from_directory('static/models/face-api', filename)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if 'user' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if verify_credentials(username, password):
            session['user'] = username
            
            # Redirect to original destination if it exists
            next_url = session.pop('next_url', None)
            if next_url:
                return redirect(next_url)
            
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    global video_stream
    
    # Stop any active streams
    if video_stream is not None:
        video_stream.stop()
        video_stream = None
        
    session.pop('user', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Main dashboard page"""
    persons = db.get_all_persons()
    recent_detections = db.get_recent_detections(10)
    active_guests = db.get_active_guests()
    
    # Format detection timestamps - convert sqlite3.Row objects to dictionaries
    formatted_detections = []
    for detection in recent_detections:
        # Convert Row to dict
        detection_dict = dict(detection)
        # Add formatted time
        if 'timestamp' in detection_dict:
            detection_dict['formatted_time'] = datetime.strptime(detection_dict['timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %I:%M %p")
        formatted_detections.append(detection_dict)
    
    # Get training status
    training_status = is_training
    
    # Get current detection interval
    detection_interval = app.config['DETECTION_INTERVAL']
    
    return render_template('index.html', 
                           persons=persons, 
                           recent_detections=formatted_detections,
                           active_guests=active_guests,
                           is_detecting=is_detecting,
                           is_training=training_status,
                           detection_interval=detection_interval)

# @app.route('/start_detection')
# @login_required
# def start_detection():
#     """Start face detection"""
#     global video_stream, is_detecting, recent_detections_cache
    
#     if video_stream is None or is_detecting is False:
#         camera_source = int(db.get_setting('camera_source') or 0)
#         detection_interval = int(db.get_setting('detection_interval') or 2000)  # milliseconds
        
#         # Reset the detection cache
#         recent_detections_cache = {}
        
#         video_stream = FaceDetectionStream(
#             face_recognizer=face_recognition_system,
#             db=db,
#             alert_system=alert_system,
#             src=camera_source,
#             detection_interval=detection_interval,
#             frames_per_detection=app.config['FRAMES_PER_DETECTION']
#         ).start()
        
#         is_detecting = True
#         time.sleep(1)  # Allow time for camera to start
    
#     return redirect(url_for('index'))

# @app.route('/stop_detection')
# @login_required
# def stop_detection():
#     """Stop face detection"""
#     global video_stream, is_detecting
    
#     if video_stream is not None:
#         video_stream.stop()
#         video_stream = None
#         is_detecting = False
    
#     return redirect(url_for('index'))

# def generate_frames():
#     """Generate video frames for streaming"""
#     global video_stream, recent_detections_cache
    
#     # Keep track of the last update time for detection results
#     last_detection_time = 0
#     detection_interval = app.config['DETECTION_INTERVAL']  # in seconds
    
#     while True:
#         if video_stream is None or video_stream.read() is None:
#             # If no stream, return a placeholder frame
#             placeholder = cv2.imread('static/images/placeholder.jpg') if os.path.exists('static/images/placeholder.jpg') else None
#             if placeholder is None:
#                 # Create a black frame with text if no placeholder exists
#                 placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
#                 cv2.putText(placeholder, "No video stream available", (50, 240), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
#             ret, buffer = cv2.imencode('.jpg', placeholder)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             time.sleep(0.1)  # Don't flood with frames
#             continue
        
#         # Get frame from stream with face detection boxes
#         frame = video_stream.read()
        
#         # Get current detected names
#         current_names = video_stream.get_detected_names()
        
#         # Check if it's time to process new detections
#         current_time = time.time()
#         if current_time - last_detection_time >= detection_interval and current_names:
#             last_detection_time = current_time
            
#             # Process each detected face (but avoid duplicate entries)
#             for name in current_names:
#                 # Skip if this name was detected recently (within the cache period)
#                 cache_key = f"{name}_{datetime.now().strftime('%Y-%m-%d_%H_%M')}"
#                 if cache_key in recent_detections_cache:
#                     continue
                
#                 # Add to cache with expiry
#                 recent_detections_cache[cache_key] = current_time + (detection_interval * 3)
                
#                 # Clean up expired cache entries
#                 expired_keys = [k for k, v in recent_detections_cache.items() if v < current_time]
#                 for key in expired_keys:
#                     recent_detections_cache.pop(key, None)
        
#         # Convert to JPEG (optimize for smoother streaming)
#         encoding_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
#         ret, buffer = cv2.imencode('.jpg', frame, encoding_params)
#         frame = buffer.tobytes()
        
#         # Return the frame in the HTTP response
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
#         # Add a small delay to control frame rate and CPU usage
#         time.sleep(0.03)  # ~30fps



# API endpoint for frontend face recognition with time-based log filtering
@app.route('/api/recognize_face', methods=['POST'])
@login_required
def api_recognize_face():
    """API endpoint for recognizing faces sent from frontend"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty file provided'}), 400
    
    try:
        # Read the image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        # Process with face recognition
        processed_frame, names, face_locations = face_recognition_system.recognize_faces(image)
        print(names)
        
        # Get database connection
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Generate response data
        detections = []
        for i, name in enumerate(names):
            # Get person status (known/guest/unknown)
            status = 'unknown'
            person_id = None
            was_logged = False
            
            if name != "Unknown":
                # Get person ID
                person = db.get_person_by_name(name)
                if person:
                    person_id = person['id']

                    current_date = date.today().isoformat()  # 'YYYY-MM-DD' format

                    cursor.execute("""
                        SELECT id FROM GUEST 
                        WHERE person_id = ? AND ? BETWEEN start_date AND end_date
                    """, (person_id, current_date))

                    guest = cursor.fetchone()
                    
                    status = 'guest' if guest else 'recognized'
                    
                    # Check if this person was recently logged (within 30 seconds)
                    cursor.execute("""
                        SELECT id, datetime(timestamp) as log_time 
                        FROM DETECTION
                        WHERE person_id = ? 
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """, (person_id,))
                    
                    recent_logs = cursor.fetchall()
                    
                    # Check if any logs are within the last 30 seconds
                    current_time = datetime.now()
                    recently_logged = False
                    
                    for log in recent_logs:
                        log_time = datetime.strptime(log['log_time'], '%Y-%m-%d %H:%M:%S')
                        time_diff = (current_time - log_time).total_seconds()
                        
                        if time_diff < 30:  # Within 30 seconds
                            recently_logged = True
                            break
                    
                    if not recently_logged:
                        # Log the detection only if not recently logged
                        db.log_detection(person_id, status, None)
                        was_logged = True
                    else:
                        print(f"Skipping log for {name} - detected within last 30 seconds")
            else:
                # Handle unknown face - more strict timing (10 seconds)
                # First check if there are recent unknown detections
                cursor.execute("""
                    SELECT id, datetime(timestamp) as log_time 
                    FROM DETECTION
                    WHERE person_id IS NULL AND status = 'unknown'
                    ORDER BY timestamp DESC
                    LIMIT 5
                """)
                
                recent_unknown_logs = cursor.fetchall()
                current_time = datetime.now()
                recently_logged_unknown = False
                
                for log in recent_unknown_logs:
                    log_time = datetime.strptime(log['log_time'], '%Y-%m-%d %H:%M:%S')
                    time_diff = (current_time - log_time).total_seconds()
                    print('time_diff',time_diff)
                    
                    if time_diff < 20:  # Within 10 seconds for unknown
                        recently_logged_unknown = True
                        break
                
                if not recently_logged_unknown and i < len(face_locations) and face_locations[i] is not None:
                    # Save unknown face image
                    face_img_path = face_recognition_system.save_unknown_face(image, face_locations[i])
                    
                    # Log detection and send alert
                    detection_id = db.log_detection(None, "unknown", face_img_path)
                    was_logged = True
                    
                    # Check if alert should be sent
                    if db.get_setting('alert_for_unknown') == 'true':
                        # alert_system.send_email_alert(detection_id, face_img_path, "Unknown")
                        trigger_alert(detection_id, face_img_path)
                else:
                    print("Skipping log for unknown face - detected within last 10 seconds")
            
            # Add to response
            detections.append({
                'name': name,
                'status': status,
                'person_id': person_id,
                'was_logged': was_logged,
                'box': face_locations[i] if i < len(face_locations) else None
            })
        
        # Return recognition results
        return jsonify({
            'success': True,
            'detections': detections
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    

import threading

def trigger_alert(detection_id, face_img_path):
    def send_alert():
        alert_system.send_email_alert(detection_id, face_img_path, "Unknown")
    threading.Thread(target=send_alert).start()


# API Routes for AJAX updates
@app.route('/api/get_recent_detections')
@login_required
def api_get_recent_detections():
    """API endpoint to get recent detections"""
    recent_detections = db.get_recent_detections(10)
    
    # Format detection timestamps - convert sqlite3.Row objects to dictionaries
    formatted_detections = []
    for detection in recent_detections:
        # Convert Row to dict
        detection_dict = dict(detection)
        
        # Add formatted time
        if 'timestamp' in detection_dict:
            try:
                detection_dict['formatted_time'] = datetime.strptime(
                    detection_dict['timestamp'], 
                    "%Y-%m-%d %H:%M:%S"
                ).strftime("%b %d, %I:%M %p")
            except ValueError:
                # Handle potential date format issues
                detection_dict['formatted_time'] = detection_dict['timestamp']
        
        formatted_detections.append(detection_dict)
    
    return jsonify(formatted_detections)

@app.route('/api/get_active_guests')
@login_required
def api_get_active_guests():
    """API endpoint to get active guests"""
    active_guests = db.get_active_guests()
    
    # Convert sqlite3.Row objects to dictionaries
    guests_list = []
    for guest in active_guests:
        # Convert Row to dict
        guest_dict = dict(guest)
        guests_list.append(guest_dict)
    
    return jsonify(guests_list)

@app.route('/api/toggle_face_boxes', methods=['POST'])
@login_required
def api_toggle_face_boxes():
    """API endpoint to toggle showing face boxes"""
    data = request.json
    show_boxes = data.get('show_boxes', True)
    
    # Store in session
    session['show_face_boxes'] = show_boxes
    
    return jsonify({'success': True})

# Enhanced AJAX versions of start/stop detection endpoints
@app.route('/start_detection')
@login_required
def start_detection():
    """Start face detection"""
    global video_stream, is_detecting, recent_detections_cache
    
    if video_stream is None or is_detecting is False:
        # camera_source = int(db.get_setting('camera_source') or 0)
        try:
            camera_source = int(db.get_setting('camera_source') or 0)
            # Test if the camera can be opened
            test_cap = cv2.VideoCapture(camera_source)
            if not test_cap.isOpened():
                print(f"Warning: Could not open camera at index {camera_source}, trying index 0")
                camera_source = 0
            test_cap.release()
        except Exception as e:
            print(f"Error accessing camera: {e}")
            camera_source = 0  # Fallback to default camera
        detection_interval = int(db.get_setting('detection_interval') or 2000)  # milliseconds
        frames_per_detection = int(app.config['FRAMES_PER_DETECTION'])
        
        # Reset the detection cache
        recent_detections_cache = {}
        
        # Use the improved dual stream system
        video_stream = DualStreamSystem(
            face_recognizer=face_recognition_system,
            db=db,
            alert_system=alert_system,
            src=camera_source,
            detection_interval=app.config['DETECTION_INTERVAL'],
            frames_per_detection=frames_per_detection
        ).start()
        
        is_detecting = True
        time.sleep(1)  # Allow time for camera to start
    
    # Check if it's an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})
    else:
        return redirect(url_for('index'))

@app.route('/stop_detection')
@login_required
def stop_detection():
    """Stop face detection"""
    global video_stream, is_detecting
    
    if video_stream is not None:
        video_stream.stop()
        video_stream = None
        is_detecting = False
    
    # Check if it's an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})
    else:
        return redirect(url_for('index'))
    

def generate_frames():
    """Generate video frames for streaming"""
    global video_stream
    
    # Copy session value to avoid context issues
    show_boxes = True  # Default value
    
    # Get session value if in request context
    if has_request_context():
        show_boxes = session.get('show_face_boxes', True)
    
    last_frame_time = time.time()
    frame_count = 0
    target_fps = 30.0
    frame_time = 1.0 / target_fps
    
    # Create a placeholder frame for when no stream is available
    placeholder = None
    
    while True:
        # Calculate time since last frame to control frame rate
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        # If we're sending frames too quickly, wait to maintain target FPS
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        
        # Check if stream is available
        if video_stream is None or video_stream.read() is None:
            # If no stream, return a placeholder frame
            if placeholder is None:
                placeholder_path = 'static/images/placeholder.jpg'
                if os.path.exists(placeholder_path):
                    placeholder = cv2.imread(placeholder_path)
                else:
                    # Create a black frame with text if no placeholder exists
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No video stream available", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Compress and send placeholder frame
            ret, buffer = cv2.imencode('.jpg', placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Sleep longer when no stream to reduce CPU usage
            time.sleep(0.5)
            continue
        
        # Get frame from stream (this now uses the dual-stream system for smoother video)
        frame = video_stream.read()
        
        # Check if we should show face boxes (outside of session to avoid context issues)
        if not show_boxes and hasattr(video_stream, 'raw_frame') and video_stream.raw_frame is not None:
            frame = video_stream.raw_frame.copy()
        
        # Add timestamp to the frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Optimize JPEG compression for faster streaming
        encoding_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', frame, encoding_params)
        frame = buffer.tobytes()
        
        # Return the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Update frame timing for FPS control
        last_frame_time = time.time()
        frame_count += 1

            
@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/users')
@login_required
def users():
    """User management page"""
    persons = db.get_all_persons()
    
    # Get minimum required face images
    min_face_images = app.config['MIN_FACE_IMAGES']
    
    # Get training status
    training_status = is_training
    
    return render_template('users.html', 
                           persons=persons, 
                           min_face_images=min_face_images,
                           is_training=training_status)

@app.route('/add_user', methods=['GET', 'POST'])
@login_required
def add_user():
    """Add new user page"""
    if request.method == 'POST':
        name = request.form.get('name')
        gender = request.form.get('gender')
        dob = request.form.get('dob')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        
        # Validate required fields
        if not name:
            flash('Name is required', 'danger')
            return redirect(url_for('add_user'))
        
        # Check if user already exists
        existing_user = db.get_person_by_name(name)
        if existing_user:
            flash(f'User {name} already exists', 'danger')
            return redirect(url_for('add_user'))
        
        # Add person to database
        person_id = db.add_person(name, gender, dob, mobile, email)
        
        # Create directory for user
        os.makedirs(f"static/images/users/{name}", exist_ok=True)
        
        flash(f'User {name} added successfully. Now add face images.', 'success')
        return redirect(url_for('add_face', person_id=person_id))
    
    return render_template('add_user.html')

@app.route('/add_face/<int:person_id>', methods=['GET', 'POST'])
@login_required
def add_face(person_id):
    """Add face images to a user"""
    person = db.get_person(person_id)
    
    if not person:
        flash('User not found', 'danger')
        return redirect(url_for('users'))
    
    if request.method == 'POST':
        # Check if it's a training request
        if 'train_model' in request.form:
            # Request model training
            if not is_training:
                training_queue.put("TRAIN")
                flash('Face recognition model training started in the background', 'info')
            else:
                flash('Model training is already in progress', 'warning')
            return redirect(url_for('add_face', person_id=person_id))
            
        # Check if the post request has the file part
        if 'face_images' not in request.files:
            flash('No face images uploaded', 'danger')
            return redirect(request.url)
        
        files = request.files.getlist('face_images')
        
        if len(files) < 1:
            flash('Please upload at least one face image', 'danger')
            return redirect(request.url)
        
        # Save each uploaded image
        saved_count = 0
        for i, file in enumerate(files):
            if file and file.filename:
                # Count existing files to determine next index
                person_dir = f"static/images/users/{person['name']}"
                existing_files = os.listdir(person_dir)
                next_index = len(existing_files) + 1
                
                filename = secure_filename(f"{person['name']}_{next_index}.jpg")
                file_path = os.path.join(person_dir, filename)
                file.save(file_path)
                
                # Add face to database
                db.add_face(person_id, file_path.replace("\\","/"))
                saved_count += 1
        
        flash(f'{saved_count} face images added successfully for {person["name"]}', 'success')
        
        # Get updated face count
        faces = db.get_faces(person_id)
        
        # Check if we have the minimum required images and should train
        if len(faces) >= app.config['MIN_FACE_IMAGES'] and not is_training:
            # User needs to click train button
            flash(f'You now have {len(faces)} face images. Click "Train Model" to update the recognition model.', 'info')
    
    # Get existing faces
    faces = db.get_faces(person_id)
    
    # Get minimum required face images
    min_face_images = app.config['MIN_FACE_IMAGES']
    
    return render_template('add_face.html', 
                           person=person, 
                           faces=faces, 
                           min_face_images=min_face_images,
                           is_training=is_training)

def capture_admin_camera():
    """Capture frames from admin camera for user registration"""
    global admin_camera
    
    while admin_camera and not admin_camera.stopped:
        # Read frame
        frame = admin_camera.read()
        
        if frame is None:
            time.sleep(0.1)
            continue
            
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add a small delay to control frame rate
        time.sleep(0.03)  # ~30fps

@app.route('/admin_camera_feed')
@login_required
def admin_camera_feed():
    """Admin camera feed for user registration"""
    return Response(capture_admin_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_admin_camera')
@login_required
def start_admin_camera():
    """Start the admin camera for face capture"""
    global admin_camera
    
    # Check if camera is already running
    if admin_camera is None or admin_camera.stopped:
        # Get camera source
        camera_source = int(db.get_setting('camera_source') or 0)
        admin_camera = VideoStream(src=camera_source).start()
        time.sleep(1)  # Allow camera to warm up
    
    return jsonify({'success': True})

@app.route('/stop_admin_camera')
@login_required
def stop_admin_camera():
    """Stop the admin camera"""
    global admin_camera
    
    if admin_camera is not None:
        admin_camera.stop()
        admin_camera = None
    
    return jsonify({'success': True})

@app.route('/capture_face/<int:person_id>', methods=['POST'])
@login_required
def capture_face(person_id):
    """Capture face image from webcam"""
    person = db.get_person(person_id)
    
    if not person:
        return jsonify({'success': False, 'message': 'User not found'})
    
    # Get image data from AJAX request
    image_data = request.json.get('image_data')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'No image data received'})
    
    # Remove the base64 prefix
    image_data = image_data.replace('data:image/jpeg;base64,', '')
    
    # Convert base64 to image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Generate filename
    # Count existing files to determine next index
    person_dir = f"static/images/users/{person['name']}"
    os.makedirs(person_dir, exist_ok=True)
    existing_files = os.listdir(person_dir)
    next_index = len(existing_files) + 1
    
    filename = f"{person['name']}_{next_index}.jpg"
    file_path = os.path.join(person_dir, filename)
    
    # Save image
    image.save(file_path)
    
    # Add face to database
    db.add_face(person_id, file_path.replace("\\",'/'))
    
    # Get updated face count
    faces = db.get_faces(person_id)
    
    return jsonify({
        'success': True, 
        'message': 'Face captured successfully',
        'file_path': "/"+ file_path.replace("\\",'/'),
        'face_count': len(faces),
        'min_required': app.config['MIN_FACE_IMAGES']
    })

@app.route('/get_face_count/<int:person_id>')
@login_required
def get_face_count(person_id):
    """Get the number of face images for a person"""
    faces = db.get_faces(person_id)
    return jsonify({
        'count': len(faces),
        'min_required': app.config['MIN_FACE_IMAGES']
    })

@app.route('/delete_user/<int:person_id>', methods=['POST'])
@login_required
def delete_user(person_id):
    """Delete a user"""
    person_name = db.delete_person(person_id)
    
    if person_name:
        # Delete user directory
        user_dir = f"static/images/users/{person_name}"
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
        
        # Request model training
        if not is_training:
            training_queue.put("TRAIN")
            flash(f'User {person_name} deleted successfully. Face recognition model retraining started', 'success')
        else:
            flash(f'User {person_name} deleted successfully. Please manually retrain the model later.', 'success')
    else:
        flash('User not found', 'danger')
    
    return redirect(url_for('users'))

@app.route('/train_model', methods=['POST'])
@login_required
def train_model():
    """Manually train the face recognition model"""
    if is_training:
        flash('Model training is already in progress', 'warning')
    else:
        training_queue.put("TRAIN")
        flash('Face recognition model training started in the background', 'info')
    
    # Redirect back to the referrer
    referrer = request.referrer or url_for('index')
    return redirect(referrer)

@app.route('/guests')
@login_required
def guests():
    """Guest management page"""
    guests = db.get_all_guests()
    persons = db.get_all_persons()
    return render_template('guests.html', guests=guests, persons=persons)

@app.route('/add_guest', methods=['POST'])
@login_required
def add_guest():
    """Add a guest"""
    person_id = request.form.get('person_id')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    purpose = request.form.get('purpose')
    
    # Validate required fields
    if not person_id or not start_date or not end_date:
        flash('Person, start date, and end date are required', 'danger')
        return redirect(url_for('guests'))
    
    # Add guest to database
    db.add_guest(person_id, start_date, end_date, purpose)
    
    flash('Guest added successfully', 'success')
    return redirect(url_for('guests'))

@app.route('/delete_guest/<int:guest_id>', methods=['POST'])
@login_required
def delete_guest(guest_id):
    """Delete a guest"""
    success = db.delete_guest(guest_id)
    
    if success:
        flash('Guest deleted successfully', 'success')
    else:
        flash('Guest not found', 'danger')
    
    return redirect(url_for('guests'))

@app.route('/alerts')
@login_required
def alerts():
    """Alert history page"""
    # Get detections with unknown status
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT d.*, p.name, a.sent_at, a.alert_type, a.status as alert_status
    FROM DETECTION d
    LEFT JOIN PERSON p ON d.person_id = p.id
    LEFT JOIN ALERT a ON d.id = a.detection_id
    WHERE d.status = 'unknown'
    ORDER BY d.timestamp DESC
    LIMIT 50
    ''')
    
    alerts = cursor.fetchall()
    
    return render_template('alerts.html', alerts=alerts)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Settings page"""
    if request.method == 'POST':
        # Update settings
        email_recipient = request.form.get('email_recipient')
        alert_for_unknown = request.form.get('alert_for_unknown') == 'on'
        camera_source = request.form.get('camera_source')
        detection_interval = request.form.get('detection_interval')
        min_detection_confidence = request.form.get('min_detection_confidence')
        min_face_images = request.form.get('min_face_images')
        detection_interval_seconds = request.form.get('detection_interval_seconds')
        frames_per_detection = request.form.get('frames_per_detection')
        
        # Update in database
        db.update_setting('email_recipient', email_recipient)
        db.update_setting('alert_for_unknown', 'true' if alert_for_unknown else 'false')
        db.update_setting('camera_source', camera_source)
        db.update_setting('detection_interval', detection_interval)
        db.update_setting('min_detection_confidence', min_detection_confidence)
        
        # Update app configurations
        if min_face_images and min_face_images.isdigit():
            app.config['MIN_FACE_IMAGES'] = int(min_face_images)
        
        if detection_interval_seconds and detection_interval_seconds.isdigit():
            app.config['DETECTION_INTERVAL'] = int(detection_interval_seconds)
        
        if frames_per_detection and frames_per_detection.isdigit():
            app.config['FRAMES_PER_DETECTION'] = int(frames_per_detection)
        
        # Update in alert system
        alert_system.email_recipient = email_recipient
        alert_system.alert_for_unknown = alert_for_unknown
        
        # Update face recognition tolerance
        face_recognition_system.tolerance = float(min_detection_confidence)
        
        flash('Settings updated successfully', 'success')
        return redirect(url_for('settings'))
    
    # Get current settings
    settings = {}
    all_settings = db.get_all_settings()
    
    for setting in all_settings:
        settings[setting['setting_name']] = setting['value']
    
    # Add app configurations
    settings['min_face_images'] = app.config['MIN_FACE_IMAGES']
    settings['detection_interval_seconds'] = app.config['DETECTION_INTERVAL']
    settings['frames_per_detection'] = app.config['FRAMES_PER_DETECTION']
    
    return render_template('settings.html', settings=settings)

@app.route('/change_password', methods=['POST'])
@login_required
def change_admin_password():
    """Change admin password"""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Verify current credentials
    if not verify_credentials(session['user'], current_password):
        flash('Current password is incorrect', 'danger')
        return redirect(url_for('settings'))
    
    # Validate new password
    if not new_password or len(new_password) < 6:
        flash('New password must be at least 6 characters', 'danger')
        return redirect(url_for('settings'))
    
    # Confirm passwords match
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
        return redirect(url_for('settings'))
    
    # Change password
    success = change_password(session['user'], new_password)
    
    if success:
        flash('Password changed successfully', 'success')
    else:
        flash('Failed to change password', 'danger')
    
    return redirect(url_for('settings'))

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    """Generate alert report"""
    date = request.form.get('report_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Get connection and cursor
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get all alerts for the specified date
    cursor.execute('''
    SELECT d.id, d.timestamp, d.status, d.image_path, 
           p.name, p.gender, p.mobile, p.email,
           a.sent_at, a.alert_type, a.status as alert_status
    FROM DETECTION d
    LEFT JOIN PERSON p ON d.person_id = p.id
    LEFT JOIN ALERT a ON d.id = a.detection_id
    WHERE date(d.timestamp) = ?
    ORDER BY d.timestamp DESC
    ''', (date,))
    
    alerts = cursor.fetchall()
    
    # Generate CSV
    import csv
    from io import StringIO
    
    output = StringIO()
    fieldnames = ['ID', 'Timestamp', 'Name', 'Status', 'Gender', 'Mobile', 'Email', 
                 'Alert Sent', 'Alert Type', 'Alert Status']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for alert in alerts:
        writer.writerow({
            'ID': alert['id'],
            'Timestamp': alert['timestamp'],
            'Name': alert['name'] if alert['name'] else 'Unknown',
            'Status': alert['status'],
            'Gender': alert['gender'] if alert['gender'] else 'N/A',
            'Mobile': alert['mobile'] if alert['mobile'] else 'N/A',
            'Email': alert['email'] if alert['email'] else 'N/A',
            'Alert Sent': alert['sent_at'] if alert['sent_at'] else 'No',
            'Alert Type': alert['alert_type'] if alert['alert_type'] else 'N/A',
            'Alert Status': alert['alert_status'] if alert['alert_status'] else 'N/A'
        })
    
    # Create response
    from flask import make_response
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename=alert_report_{date}.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# Cleanup function
def cleanup():
    """Clean up resources on shutdown"""
    global video_stream, training_thread, admin_camera
    
    # Stop the video stream
    if video_stream is not None:
        video_stream.stop()
    
    # Stop the admin camera
    if admin_camera is not None:
        admin_camera.stop()
    
    # Stop the training thread
    if training_thread is not None and training_thread.is_alive():
        training_queue.put("EXIT")
        training_thread.join(timeout=1)

# Register cleanup function
import atexit
atexit.register(cleanup)

# Run the Flask app
if __name__ == '__main__':
    import numpy as np  # For placeholder frame
    app.run(debug=True, host='0.0.0.0', threaded=True)