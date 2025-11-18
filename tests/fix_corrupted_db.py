#!/usr/bin/env python3
"""
Fix corrupted database - Two-table structure only.
"""
import os
import shutil
import sqlite3
from datetime import datetime

def fix_database():
    """Backup corrupted DB and create fresh one."""
    
    db_file = "data/drowsiness_events.db"
    
    print("=" * 60)
    print("Database Recovery Tool")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(db_file):
        print(f"\n❌ Database file not found: {db_file}")
        print("   Creating new database...")
        create_fresh_database(db_file)
        return
    
    # Check if file is valid SQLite database
    print(f"\nChecking database: {db_file}")
    
    try:
        with sqlite3.connect(db_file) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"✓ Valid database with tables: {tables}")
            
            # Remove users table if it exists (we don't need local cache)
            if 'users' in tables:
                print("\n⚠️  Removing 'users' table (user data stored on server only)...")
                conn.execute("DROP TABLE users")
                conn.commit()
                print("✓ users table removed")
            
            # Add missing user_profiles table if needed
            if 'user_profiles' not in tables:
                print("\n⚠️  user_profiles table missing. Creating it...")
                conn.execute("""
                    CREATE TABLE user_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        ear_threshold REAL NOT NULL,
                        face_encoding BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX idx_user_id ON user_profiles(user_id)")
                conn.commit()
                print("✓ user_profiles table created")
            
            print("\n✅ Database is healthy!")
            return
    
    except sqlite3.DatabaseError as e:
        print(f"❌ Database is corrupted: {e}")
        
        # Backup corrupted file
        backup_file = db_file.replace('.db', f'_corrupted_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        print(f"\n📦 Creating backup: {backup_file}")
        shutil.copy2(db_file, backup_file)
        
        # Remove corrupted file
        os.remove(db_file)
        print(f"✓ Removed corrupted file")
        
        # Create fresh database
        create_fresh_database(db_file)
        
        print(f"\n✅ Database recreated successfully!")
        print(f"   Backup saved: {backup_file}")

def create_fresh_database(db_file):
    """Create a fresh database with only necessary tables."""
    
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    with sqlite3.connect(db_file) as conn:
        # Create drowsiness_events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drowsiness_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_identification_number TEXT NOT NULL,
                user_id INTEGER,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                details TEXT,
                img_drowsiness BLOB,
                img_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_profiles table (NO users table - data on server only)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ear_threshold REAL NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_profiles(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_time ON drowsiness_events(time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_user ON drowsiness_events(user_id)")
        
        conn.commit()
    
    print(f"✓ Fresh database created: {db_file}")
    print(f"  Tables: drowsiness_events, user_profiles")
    print(f"  Note: User personal data stored on server only")

if __name__ == "__main__":
    fix_database()