import socket

# --- Configuration ---
# IMPORTANT: Change this to the actual local IP address of your PC!
PC_HOST = '10.175.189.123' # Example: '192.168.1.100'
# This port MUST be the exact same one the PC server is using.
PC_PORT = 65432

# The message you want to send
message_to_send = '1'

print("--- Raspberry Pi Client (Publisher) ---")

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Attempting to connect to PC server at {PC_HOST}:{PC_PORT}...")
        
        # Connect to the server
        s.connect((PC_HOST, PC_PORT))
        
        print("Connection successful!")
        print(f"Sending message: '{message_to_send}'")
        
        # Encode the message into bytes and send it
        s.sendall(message_to_send.encode('utf-8'))
        
        print("Message sent successfully.")

except ConnectionRefusedError:
    print(f"\n[ERROR] Connection refused. Please check that:")
    print(f"1. The 'pc_server.py' script is running on the PC.")
    print(f"2. The IP address '{PC_HOST}' is correct for your PC.")
    print(f"3. Both devices are on the same Wi-Fi network.")
except socket.gaierror:
    print(f"\n[ERROR] Hostname could not be resolved. Is the IP address '{PC_HOST}' correct?")
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")

print("Client script finished.")