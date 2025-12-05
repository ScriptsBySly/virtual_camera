import socket
import threading

# ---------------- CONFIG ----------------
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000       # Port to listen on

# ---------------- HANDLE CLIENT ----------------
def handle_client(client_socket, address):
    print(f"New connection from {address}")
    with client_socket:
        while True:
            try:
                data = client_socket.recv(1024)  # Receive up to 1024 bytes
                if not data:
                    break
                print(f"[{address}] Received: {data.decode('utf-8')}")
            except ConnectionResetError:
                break
    print(f"Connection closed: {address}")

# ---------------- SERVER ----------------
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        client_socket, address = server.accept()
        thread = threading.Thread(target=handle_client, args=(client_socket, address), daemon=True)
        thread.start()

if __name__ == "__main__":
    start_server()
