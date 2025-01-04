import socket
import threading
import subprocess
import open3d as o3d
import numpy as np


def process_pose_data(data):
    """Process the received pose data."""
    data = data.split(',')
    rotation, translation = [], []
    for i, x in enumerate(len(data)):
        if i in [3, 7, 11]:
            translation.append(float(x))
        else:
            rotation.append(float(x))
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation)
    return rotation, translation


def socket_listener(pose_stack):
    """Socket listener to receive data from the C++ client."""
    host = '127.0.0.1'
    port = 12345

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print("Listening for connections...")

    conn, addr = server.accept()
    print(f"Connected by {addr}")

    while True:
        data = conn.recv(1024)
        if not data:
            break
        pose_stack.append(data.decode('utf-8'))

    conn.close()


def visualize_pose(pose_stack):
    """Visualize poses using Open3D."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(mesh_frame)

    while True:
        if pose_stack:
            data = pose_stack.pop()
            rotation, translation = process_pose_data(data)

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            mesh_frame.rotate(rotation, center=(0, 0, 0))
            mesh_frame.translate(translation)

            vis.update_geometry(mesh_frame)
            vis.poll_events()
            vis.update_renderer()


def launch_cpp_program(vocabulary_path, settings_path, port):
    """Launch the C++ program using subprocess and pass vocabulary, settings, and port as arguments."""
    cpp_executable_path = "./stereo_inertial_realsense_D435i"  # Adjust if needed
    try:
        # Passing all three arguments: vocabulary, settings, and port
        subprocess.Popen([cpp_executable_path, vocabulary_path, settings_path, str(port)])
        print(f"Launched C++ program with port {port}.")
    except FileNotFoundError:
        print("Error: Ensure the C++ program is compiled and the path is correct.")
    except Exception as e:
        print(f"Error launching C++ program: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the Python SLAM visualizer with customizable port.")
    parser.add_argument('--port', type=int, default=12345, help='Port number for the server.')
    parser.add_argument('--vocabulary', type=str, required=True, help='Path to the vocabulary file.')
    parser.add_argument('--settings', type=str, required=True, help='Path to the settings file.')
    args = parser.parse_args()

    # Call the updated function with the provided arguments
    launch_cpp_program(args.vocabulary, args.settings, args.port)

    pose_track = []

    listener_thread = threading.Thread(target=socket_listener, args=(pose_track,))
    listener_thread.daemon = True
    listener_thread.start()

    visualize_pose(pose_track)
