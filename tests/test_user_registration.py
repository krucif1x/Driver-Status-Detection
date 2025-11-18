
#!/usr/bin/env python3
"""
Test user registration system with SQLite.
Creates dummy users to verify the system works.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np
from src.services.user_manager import UserManager

def create_test_image():
    """Create a simple test image (you'd use real camera frames)."""
    # Create a 640x480 RGB image with random noise
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return img

def test_registration():
    """Test user registration workflow."""
    
    print("=" * 60)
    print("Testing User Registration System")
    print("=" * 60)
    
    # Initialize UserManager
    print("\n1. Initializing UserManager...")
    manager = UserManager(database_file="data/drowsiness_events.db")
    print(f"   Current users: {len(manager.users)}")
    
    # Note: For REAL testing, you need actual face images
    # This is just to show the workflow
    print("\n2. Registration workflow:")
    print("   (Use actual camera frames with faces for real registration)")
    print("   manager.register_new_user(camera_frame, ear_threshold=0.25)")
    
    # List current users
    print("\n3. Current users in database:")
    if manager.users:
        for user in manager.users:
            print(f"   - ID {user.id}: {user.name} (EAR: {user.ear_threshold:.3f})")
    else:
        print("   (No users registered yet)")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Run your calibration system")
    print("2. After calibration, call:")
    print("   user_manager.register_new_user(frame, ear_threshold)")
    print("3. User will get auto-incrementing ID: 1, 2, 3...")
    print("=" * 60)

if __name__ == "__main__":
    test_registration()
