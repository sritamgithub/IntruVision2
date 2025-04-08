from flask import session, redirect, url_for, request, flash
from functools import wraps
import os
import hashlib
import sqlite3

# Default admin credentials (to be stored securely in a database)
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"  # This should be hashed in production

def init_auth_db(db_connection):
    """Initialize authentication table in the database"""
    cursor = db_connection.cursor()
    
    # Create ADMIN table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ADMIN (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Check if there's at least one admin user
    cursor.execute('SELECT COUNT(*) FROM ADMIN')
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Add default admin user
        password_hash = hashlib.sha256(DEFAULT_ADMIN_PASSWORD.encode()).hexdigest()
        
        cursor.execute('''
        INSERT INTO ADMIN (username, password_hash)
        VALUES (?, ?)
        ''', (DEFAULT_ADMIN_USERNAME, password_hash))
    
    db_connection.commit()

def verify_credentials(username, password):
    """Verify username and password against database"""
    conn = sqlite3.connect('students.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get user with matching username
    cursor.execute('SELECT * FROM ADMIN WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user is None:
        return False
    
    # Hash the provided password and compare
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == user['password_hash']

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page', 'warning')
            # Store original destination for redirect after login
            session['next_url'] = request.url
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def change_password(username, new_password):
    """Change user password"""
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    
    # Hash the new password
    password_hash = hashlib.sha256(new_password.encode()).hexdigest()
    
    # Update the password
    cursor.execute('''
    UPDATE ADMIN SET password_hash = ? WHERE username = ?
    ''', (password_hash, username))
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0