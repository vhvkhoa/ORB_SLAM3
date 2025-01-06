import socket
import threading
import subprocess
from functools import partial
import random

import open3d as o3d
import numpy as np
import keyboard
from copy import deepcopy


def find_unused_port(start_port=5000, end_port=6000, num_trials=50):
    for _ in range(num_trials):
        port = random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                print(f'Python: Selected unused port: {port}')
                return port
            except OSError:
                continue
    raise RuntimeError('Could not find an unused port in the specified range.')


def socket_listener(pose_stack, port):
    """Socket listener to receive data from the C++ client."""
    host = '127.0.0.1'

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print("Python: Listening for connections...")

    conn, addr = server.accept()
    print(f"Python: Connected by {addr}")

    while True:
        data = conn.recv(1024)
        if not data:
            break
        pose_stack.append(data.decode('utf-8'))

    conn.close()


def launch_cpp_program(vocabulary_path, settings_path, port):
    """Launch the C++ program using subprocess and pass vocabulary, settings, and port as arguments."""
    cpp_executable_path = "./Examples/Stereo-Inertial/controller_rs_D435i"  # Adjust if needed
    try:
        # Passing all three arguments: vocabulary, settings, and port
        subprocess.Popen([cpp_executable_path, vocabulary_path, settings_path, str(port)])
        print(f"Python: Launched C++ program with port {port}.")
    except FileNotFoundError:
        print("Python: Error: Ensure the C++ program is compiled and the path is correct.")
    except Exception as e:
        print(f"Python: Error launching C++ program: {e}")


def invert_transformation(transform):
    inv_transform = np.eye(4)
    inv_transform[:3, :3] = transform[:3, :3].T
    inv_transform[:3, 3:] = - transform[:3, :3].T @ transform[:3, 3:]
    return inv_transform


class RealsenseController:
    def __init__(self, args):
        self.pose_track = []
        unused_port = find_unused_port()
        listener_thread = threading.Thread(target=partial(socket_listener, port=unused_port), args=(self.pose_track,))
        listener_thread.daemon = True
        listener_thread.start()

        # Call the updated function with the provided arguments
        launch_cpp_program(args.vocabulary, args.settings, unused_port)

        self.Tw = np.eye(4)
        self.Twc = np.eye(4)
        keyboard.on_press_key('r', self.reset_reference_frame)
        self.visualize_pose()


    def process_pose_data(self, data):
        """Process the received pose data."""
        data = data.split(',')
        rotation, translation = [], []
        for i, x in enumerate(data):
            if i in [3, 7, 11]:
                translation.append(float(x))
            else:
                rotation.append(float(x))
        rotation = np.array(rotation).reshape(3, 3)
        translation = np.array(translation)

        Tcw = np.eye(4)
        Tcw[:3, :3] = rotation
        Tcw[:3, 3] = translation
        Twc = invert_transformation(Tcw) 

        return Twc

    def reset_reference_frame(self, _):
        print('Reset reference frame.')
        self.Tw = deepcopy(self.Twc)

    def visualize_pose(self):
        """Visualize poses using Open3D."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1080, height=720)
        pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480,
            fx=393.83372195014,
            fy=393.5142578625567,
            cx=323.41500704616675,
            cy=233.14763954717986,
        )
        camera = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=pinhole_intrinsic, extrinsic=np.eye(4), scale=1.0,
        )
        camera_points = deepcopy(camera.points)
        self.vis.add_geometry(camera)

        while True:
            if self.pose_track:
                self.Twc = self.process_pose_data(self.pose_track.pop())
                camera.points = camera_points
                Twc =  invert_transformation(self.Tw) @ self.Twc
                camera.transform(Twc)

                self.vis.update_geometry(camera)
                self.vis.poll_events()
                self.vis.update_renderer()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the Python SLAM visualizer with customizable port.")
    parser.add_argument('--vocabulary', type=str, required=True, help='Path to the vocabulary file.')
    parser.add_argument('--settings', type=str, required=True, help='Path to the settings file.')
    args = parser.parse_args()

    controller = RealsenseController(args)
