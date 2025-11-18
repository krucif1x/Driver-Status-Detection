
#!/usr/bin/env python3
"""
Initialize user_profiles table in drowsiness_events.db
Run this once to set up the database schema.
"""
import sqlite3
import os

def init_user_profiles_table():
    """Create user_profiles table with auto-incrementing IDs."""
    
    db_file = "data/drowsiness_events.db"
    
    print("=" * 60)
    print("Initializing User Profiles Table")
    print("=" * 60)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    # Connect and create table
    with sqlite3.connect(db_file) as conn:
        # Create user_profiles table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                ear_threshold REAL NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for fast lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_name ON user_profiles(name)")
        
        conn.commit()
    
    print(f"✓ Table created in: {db_file}")
    
    # Verify
    with sqlite3.connect(db_file) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM user_profiles")
        count = cursor.fetchone()[0]
        print(f"✓ Current users in database: {count}")
    
    print("\n" + "=" * 60)
    print("✅ Initialization Complete!")
    print("=" * 60)
    print("\nYour system is ready to register users.")
    print("Users will be assigned IDs: 1, 2, 3, 4, 5...")
    print("=" * 60)

if __name__ == "__main__":
    init_user_profiles_table()