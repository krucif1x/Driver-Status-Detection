import socket
import time

class TCPPublisher:
    def __init__(self, server_ip, server_port):
        """
        Initializes the publisher with the server's address.
        It does NOT connect at startup to prevent freezing the main application.
        """
        self.server_ip = server_ip
        self.server_port = server_port
        print(f"TCP Publisher is configured to send messages to {self.server_ip}:{self.server_port}")

    def send_message(self, message):
        """
        Establishes a connection, sends a single message, and then closes.
        This is a robust way to send state updates without holding a constant connection.
        
        Args:
            message (str): The string message to send (e.g., '1' or '0').
            
        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        try:
            # Create a new socket for each message to ensure a clean connection
            # The 'with' statement guarantees the socket is closed automatically
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set a short timeout (e.g., 1 second) to prevent long freezes
                s.settimeout(1.0)
                
                # Connect to the server
                s.connect((self.server_ip, self.server_port))
                
                # Encode the string message into bytes and send it
                s.sendall(message.encode('utf-8'))
            
            return True # Return True on success
            
        except (socket.timeout, ConnectionRefusedError):
            # These errors are expected if the server is not running.
            # We don't need to print an error every time, just return False.
            return False
        except Exception as e:
            # Catch any other unexpected network errors
            print(f"[TCP Publisher Error] An unexpected error occurred: {e}")
            return False

# This block allows you to test this script by itself
if __name__ == "__main__":
    # IMPORTANT: Use your PC's actual IP address here
    SERVER_IP = "10.175.189.123"  # Your PC's IP address
    SERVER_PORT = 65432          

    print("--- TCP Publisher Test ---")
    publisher = TCPPublisher(SERVER_IP, SERVER_PORT)

    try:
        print("\nAttempting to send '1' (Drowsy)...")
        if publisher.send_message('1'):
            print("Successfully sent '1'.")
        else:
            print("Failed to send '1'. Is the server running?")

        print("\nWaiting for 2 seconds...")
        time.sleep(2)

        print("Attempting to send '0' (Normal)...")
        if publisher.send_message('0'):
            print("Successfully sent '0'.")
        else:
            print("Failed to send '0'. Is the server running?")

    except KeyboardInterrupt:
        print("\nTest stopped by user.")