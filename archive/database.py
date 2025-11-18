import os
import numpy as np
import time
import uuid

# --- Database Management Functions (Your code, slightly polished) ---

DATABASE_FILE = os.path.join("data", "user_profiles.npz")

def load_user_database(database_file=DATABASE_FILE):
    """Loads user profiles from the NPZ file."""
    if not os.path.exists(database_file):
        print("Database not found. A new one will be created.")
        return [], [], []
    try:
        # allow_pickle=True is necessary for loading object arrays
        data = np.load(database_file, allow_pickle=True)
        names = list(data.get('names', []))
        ear_thresholds = list(data.get('ear_thresholds', []))
        face_encodings = list(data.get('face_encodings', []))
        
        # Ensure encodings are uniform (1D float arrays)
        valid_encodings = []
        if len(face_encodings) > 0:
            # Determine expected shape from the first valid encoding
            expected_shape = None
            for enc in face_encodings:
                if hasattr(enc, 'shape'):
                    expected_shape = np.array(enc).flatten().shape
                    break
            
            if expected_shape:
                for enc in face_encodings:
                    flat_enc = np.array(enc, dtype=np.float32).flatten()
                    if flat_enc.shape == expected_shape:
                        valid_encodings.append(flat_enc)
                    else:
                        print(f"Warning: Skipping malformed encoding with shape {flat_enc.shape}. Expected {expected_shape}.")

        print(f"Loaded {len(names)} user(s) from the database.")
        return names, ear_thresholds, valid_encodings
    except Exception as e:
        print(f"Error loading database file '{database_file}': {e}")
        # If loading fails, start with an empty database to prevent crashing
        return [], [], []

def save_user_database(names, ear_thresholds, face_encodings, database_file=DATABASE_FILE):
    """Saves user profiles to the NPZ file."""
    if not names:
        print("Database is empty. Nothing to save.")
        return

    os.makedirs(os.path.dirname(database_file), exist_ok=True)

    # Convert all encodings to 1D float32 arrays
    encodings_as_np = [np.array(enc, dtype=np.float32).flatten() for enc in face_encodings]

    # Filter out malformed encodings
    valid_encodings = []
    expected_shape = None
    for enc in encodings_as_np:
        if expected_shape is None:
            expected_shape = enc.shape
        if enc.shape == expected_shape:
            valid_encodings.append(enc)
        else:
            print(f"Warning: Skipping malformed encoding with shape {enc.shape}. Expected {expected_shape}.")

    # Also filter names and ear_thresholds to match valid encodings
    if len(valid_encodings) != len(names):
        print("Warning: Some encodings were malformed and not saved. Filtering names and thresholds accordingly.")
        # Only keep names and thresholds for valid encodings
        indices = [i for i, enc in enumerate(encodings_as_np) if enc.shape == expected_shape]
        names = [names[i] for i in indices]
        ear_thresholds = [ear_thresholds[i] for i in indices]

    print(f"Saving {len(names)} user(s) to the database...")
    np.savez(database_file,
             names=np.array(names),
             ear_thresholds=np.array(ear_thresholds),
             face_encodings=np.array(valid_encodings, dtype=object))
    print("Database saved successfully.")

# --- Mock Functions for Simulation ---
# In a real application, these would be your actual computer vision functions.

def get_simulated_face_encoding():
    """Simulates detecting a face and returning its encoding."""
    # This will return a new random encoding each time to simulate different people
    return np.random.rand(128).astype(np.float32)

def find_best_match(known_encodings, new_encoding, tolerance=0.6):
    """
    Simulates comparing a new face encoding against the database.
    Returns the index of the best match and the distance.
    """
    if not known_encodings:
        return None, float('inf')
        
    # Calculate Euclidean distance (L2 norm)
    distances = np.linalg.norm(np.array(known_encodings) - new_encoding, axis=1)
    
    best_match_index = np.argmin(distances)
    min_distance = distances[best_match_index]
    
    if min_distance <= tolerance:
        return best_match_index, min_distance
    else:
        return None, min_distance

# --- Main Application Logic ---

def register_new_user(new_encoding, names, ear_thresholds, face_encodings):
    """Handles the process of registering a new user."""
    print("\n--- New User Registration ---")
    name = input("Unknown face detected. Please enter your name to register: ")
    if not name:
        print("Registration cancelled.")
        return None, None

    print(f"Hello, {name}! Starting calibration process...")
    # Simulate EAR calibration
    time.sleep(2) 
    calibrated_ear = np.random.uniform(0.18, 0.28) # Simulate a calibration result
    print(f"Calibration complete! Your EAR threshold is set to: {calibrated_ear:.3f}")

    # Add new user to our lists
    names.append(name)
    ear_thresholds.append(calibrated_ear)
    face_encodings.append(new_encoding)

    # Save the updated database to disk
    save_user_database(names, ear_thresholds, face_encodings)
    
    return name, calibrated_ear

def main_loop():
    """Main application loop demonstrating the user management logic."""
    
    # 1. INITIALIZATION: Load all known users from the database on startup.
    names, ear_thresholds, face_encodings = load_user_database()

    # State variables for the current session
    current_user_name = None
    current_ear_threshold = None
    
    print("\n--- Drowsiness Detection System Activated ---")
    print("Monitoring for drivers...")

    try:
        while True:
            # In a real app, you would get a camera frame here.
            # We simulate detecting one face and getting its encoding.
            detected_encoding = get_simulated_face_encoding()

            # 2. USER IDENTIFICATION: Compare the detected face to the database.
            match_index, distance = find_best_match(face_encodings, detected_encoding)

            identified_name = None
            if match_index is not None:
                identified_name = names[match_index]

            # 3. STATE MANAGEMENT LOGIC
            if identified_name:
                # --- A KNOWN USER IS DETECTED ---
                if identified_name != current_user_name:
                    # This is the key logic: a returning user or a user swap has occurred.
                    print(f"\n✅ Welcome back, {identified_name}! (Match distance: {distance:.2f})")
                    current_user_name = identified_name
                    current_ear_threshold = ear_thresholds[match_index]
                    print(f"   Loaded profile. EAR threshold set to: {current_ear_threshold:.3f}")
                    print("   No re-calibration needed.")
            
            else:
                # --- AN UNKNOWN USER IS DETECTED ---
                if current_user_name != "Unknown":
                    # We only trigger registration once to avoid spamming the prompt.
                    print(f"\n❓ Unknown user detected. (Closest match distance: {distance:.2f})")
                    current_user_name = "Unknown" # Set state to prevent re-triggering
                    current_ear_threshold = 0.25 # Use a default threshold for now
                    
                    # Trigger the registration process
                    new_name, new_thresh = register_new_user(detected_encoding, names, ear_thresholds, face_encodings)
                    if new_name and new_thresh:
                        # If registration was successful, update the current user state immediately
                        current_user_name = new_name
                        current_ear_threshold = new_thresh

            # This is where your drowsiness detection logic would run,
            # using the `current_ear_threshold`.
            if current_user_name and current_user_name != "Unknown":
                print(f"   -> Monitoring {current_user_name} with EAR threshold {current_ear_threshold:.3f}...")
            
            time.sleep(5) # Simulate processing delay

    except KeyboardInterrupt:
        print("\n--- System shutting down. ---")
