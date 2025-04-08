import sqlite3
import os
from datetime import datetime

class Database:
    def __init__(self, db_path="students.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.create_tables()
    
    def get_connection(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create PERSON table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PERSON (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT,
            dob TEXT,
            mobile TEXT,
            email TEXT,
            user_type TEXT DEFAULT 'regular'
        )
        ''')
        
        # Create FACE table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS FACE (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES PERSON (id) ON DELETE CASCADE
        )
        ''')
        
        # Create GUEST table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS GUEST (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            purpose TEXT,
            FOREIGN KEY (person_id) REFERENCES PERSON (id) ON DELETE CASCADE
        )
        ''')
        
        # Create DETECTION table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS DETECTION (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            image_path TEXT,
            FOREIGN KEY (person_id) REFERENCES PERSON (id) ON DELETE SET NULL
        )
        ''')
        
        # Create ALERT table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ALERT (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            alert_type TEXT,
            status TEXT,
            FOREIGN KEY (detection_id) REFERENCES DETECTION (id) ON DELETE CASCADE
        )
        ''')
        
        # Create SETTINGS table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS SETTINGS (
            setting_name TEXT PRIMARY KEY,
            value TEXT,
            description TEXT
        )
        ''')
        
        # Insert default settings
        default_settings = [
            ('email_recipient', '', 'Email to receive alerts'),
            ('alert_for_unknown', 'true', 'Send alerts for unknown faces'),
            ('min_detection_confidence', '0.5', 'Minimum confidence for face detection'),
            ('camera_source', '0', 'Camera source (0 for built-in webcam)'),
            ('detection_interval', '2000', 'Interval between face detection (ms)')
        ]
        
        for setting in default_settings:
            cursor.execute('''
            INSERT OR IGNORE INTO SETTINGS (setting_name, value, description)
            VALUES (?, ?, ?)
            ''', setting)
        
        conn.commit()
    
    def add_person(self, name, gender=None, dob=None, mobile=None, email=None, user_type='regular'):
        """Add a new person to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO PERSON (name, gender, dob, mobile, email, user_type)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, gender, dob, mobile, email, user_type))
        
        conn.commit()
        return cursor.lastrowid
    
    def update_person(self, person_id, name, gender=None, dob=None, mobile=None, email=None, user_type=None):
        """Update person information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE PERSON SET name=?, gender=?, dob=?, mobile=?, email=?, user_type=?
        WHERE id=?
        ''', (name, gender, dob, mobile, email, user_type, person_id))
        
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_person(self, person_id):
        """Delete a person from the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get the person's name first (to delete their images later)
        cursor.execute('SELECT name FROM PERSON WHERE id=?', (person_id,))
        person = cursor.fetchone()
        
        if not person:
            return False
        
        cursor.execute('DELETE FROM PERSON WHERE id=?', (person_id,))
        conn.commit()
        
        return person['name']
    
    def get_person(self, person_id):
        """Get a person by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM PERSON WHERE id=?', (person_id,))
        return cursor.fetchone()
    
    def get_person_by_name(self, name):
        """Get a person by name"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM PERSON WHERE name=?', (name,))
        return cursor.fetchone()
    
    def get_all_persons(self, user_type=None):
        """Get all persons, optionally filtered by user_type"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if user_type:
            cursor.execute('SELECT * FROM PERSON WHERE user_type=?', (user_type,))
        else:
            cursor.execute('SELECT * FROM PERSON')
        
        return cursor.fetchall()
    
    def add_face(self, person_id, file_path):
        """Add a face image to a person"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO FACE (person_id, file_path)
        VALUES (?, ?)
        ''', (person_id, file_path))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_faces(self, person_id):
        """Get all faces for a person"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM FACE WHERE person_id=?', (person_id,))
        return cursor.fetchall()
    
    def add_guest(self, person_id, start_date, end_date, purpose=None):
        """Add a guest entry"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO GUEST (person_id, start_date, end_date, purpose)
        VALUES (?, ?, ?, ?)
        ''', (person_id, start_date, end_date, purpose))
        
        conn.commit()
        return cursor.lastrowid
    
    def update_guest(self, guest_id, start_date, end_date, purpose=None):
        """Update guest information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE GUEST SET start_date=?, end_date=?, purpose=?
        WHERE id=?
        ''', (start_date, end_date, purpose, guest_id))
        
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_guest(self, guest_id):
        """Delete a guest entry"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM GUEST WHERE id=?', (guest_id,))
        conn.commit()
        
        return cursor.rowcount > 0
    
    def get_active_guests(self):
        """Get all active guests (current date between start and end dates)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT g.*, p.name, p.email, p.mobile
        FROM GUEST g
        JOIN PERSON p ON g.person_id = p.id
        WHERE date('now') BETWEEN g.start_date AND g.end_date
        ''')
        
        return cursor.fetchall()
    
    def get_all_guests(self):
        """Get all guests"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT g.*, p.name, p.email, p.mobile
        FROM GUEST g
        JOIN PERSON p ON g.person_id = p.id
        ''')
        
        return cursor.fetchall()
    
    def log_detection(self, person_id, status, image_path=None):
        """Log a face detection event with explicit timestamp"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Generate current timestamp in the format SQLite expects
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
        INSERT INTO DETECTION (person_id, status, image_path, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (person_id, status, image_path, current_timestamp))
        
        conn.commit()
        return cursor.lastrowid

    def get_recent_detections(self, limit=10):
        """Get recent detection events"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT d.*, p.name
        FROM DETECTION d
        LEFT JOIN PERSON p ON d.person_id = p.id
        ORDER BY d.timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
    
    def log_alert(self, detection_id, alert_type, status):
        """Log an alert event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO ALERT (detection_id, alert_type, status)
        VALUES (?, ?, ?)
        ''', (detection_id, alert_type, status))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_setting(self, setting_name):
        """Get a setting value"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT value FROM SETTINGS WHERE setting_name=?', (setting_name,))
        setting = cursor.fetchone()
        
        return setting['value'] if setting else None
    
    def update_setting(self, setting_name, value):
        """Update a setting value"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE SETTINGS SET value=?
        WHERE setting_name=?
        ''', (value, setting_name))
        
        conn.commit()
        return cursor.rowcount > 0
    
    def get_all_settings(self):
        """Get all settings"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM SETTINGS')
        return cursor.fetchall()