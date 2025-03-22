# import threading
# from abc import abstractmethod, abstractproperty
# from typing import Dict, Optional, Union

import numpy as np
import cv2
import multiprocessing as mp
import yaml

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")


"""
NOTE: All cameras are set to record at the specified resolution using the ZED SDK.
"""


class ZEDCamera:
    def __init__(
        self,
        serial_number: str,
        width: int,
        height: int,
        use_depth: bool,
        exposure=-1,
    ):
        self.width = width
        self.height = height
        self.use_depth = use_depth
        
        # Initialize ZED camera
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA  # or sl.RESOLUTION.HD2K
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA if use_depth else sl.DEPTH_MODE.NONE
        init_params.set_from_serial_number(int(serial_number))
        
        # Add more initialization parameters
        # init_params.sdk_verbose = True  # Enable verbose logging
        init_params.sdk_gpu_id = -1  # Let SDK choose the GPU
        init_params.depth_minimum_distance = 0.3  # Meters
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        
        self.zed = sl.Camera()
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
            
        if exposure > 0:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
            
        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat() if use_depth else None

    def get_intrinsics(self):
        # Get camera information and calibration parameters
        camera_info = self.zed.get_camera_information()
        left_cam = camera_info.camera_configuration.calibration_parameters.left_cam
        return dict(
            matrix=np.array([
                [left_cam.fx, 0, left_cam.cx],
                [0, left_cam.fy, left_cam.cy],
                [0, 0, 1.0],
            ]),
            width=self.width,
            height=self.height,
            depth_scale=1.0,
        )

    def get_frames(self) -> dict[str, np.ndarray]:
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab frame from ZED camera")

        # Get color image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        image = self.image.get_data()
        image = cv2.resize(image, (self.width, self.height))
        frames = dict(image=image)

        # Get depth if enabled
        if self.use_depth:
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth = self.depth.get_data()
            depth = cv2.resize(depth, (self.width, self.height))
            frames["depth"] = depth[..., np.newaxis]

        return frames

    def close(self):
        self.zed.close()


class SequentialCameras:
    def __init__(self, camera_args_list: list[dict]):
        self.cameras: dict[str, ZEDCamera] = {}
        for camera_args in camera_args_list:
            name = camera_args.pop("name")
            self.cameras[name] = ZEDCamera(**camera_args)

    def get_intrinsics(self, name):
        return self.cameras[name].get_intrinsics()

    def get_frames(self):
        return {name: camera.get_frames() for name, camera in self.cameras.items()}

    def __del__(self):
        for camera in self.cameras.values():
            camera.close()


class ParallelCameras:
    def __init__(self, camera_args_list: list[dict]):
        # self.camera_args_list = camera_args_list
        self.name_to_camera_args = {}
        for camera_args in camera_args_list:
            name = camera_args.pop("name")
            self.name_to_camera_args[name] = camera_args

        self.camera_procs = {}
        self.put_queues = {}
        self.get_queues = {}
        self.intrinsics = {}

        for name, camera_args in self.name_to_camera_args.items():
            put_queue = mp.Queue(maxsize=1)
            get_queue = mp.Queue(maxsize=1)
            proc = mp.Process(target=self._camera_proc, args=(camera_args, put_queue, get_queue))
            proc.start()
            self.camera_procs[name] = proc

            self.intrinsics[name] = get_queue.get()
            print(f"cam {name} constructed")

            self.put_queues[name] = put_queue
            self.get_queues[name] = get_queue

    def _camera_proc(self, camera_args, receive_queue: mp.Queue, send_queue: mp.Queue):
        camera = ZEDCamera(**camera_args)
        send_queue.put(camera.get_intrinsics())

        while True:
            msg = receive_queue.get()
            if msg == "terminate":
                break

            assert msg == "get"
            assert send_queue.empty()

            frames = camera.get_frames()
            send_queue.put(frames)

        camera.close()

    def get_intrinsics(self, name):
        return self.intrinsics[name]

    def get_frames(self):
        for _, put_queue in self.put_queues.items():
            assert put_queue.empty()
            put_queue.put("get")

        camera_frames = {}
        for name, get_queue in self.get_queues.items():
            camera_frames[name] = get_queue.get()

        return camera_frames

    def __del__(self):
        # FIXME: well
        for name, put_queue in self.put_queues.items():
            print(f"terminating {name}")
            put_queue.put("terminate")
            self.camera_procs[name].join()


if __name__ == "__main__":
    # Load camera configuration from yaml file
    with open("envs/fr3.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    def test_sequential():
        print("\n=== Testing Sequential Cameras ===")
        # Get the first camera configuration and ensure it has a name
        camera_args = config["cameras"][0]  # Using the first camera from the yaml
        
        print(f"Camera constructed: {camera_args}")
        cameras = SequentialCameras([camera_args])
        if "name" not in camera_args:
            camera_args["name"] = "agent1"  # Provide a default name if none exists
        
        # Test getting intrinsics
        intrinsics = cameras.get_intrinsics(camera_args["name"])
        print("Camera intrinsics:", intrinsics)
        
        # Test getting frames
        try:
            for i in range(10):  # Get 10 frames
                frames = cameras.get_frames()
                print(f"Frame {i}:")
                for camera_name, camera_frames in frames.items():
                    print(f"  Camera {camera_name}:")
                    for key, frame in camera_frames.items():
                        print(f"    {key} shape: {frame.shape}")
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        finally:
            del cameras

    def test_parallel():
        print("\n=== Testing Parallel Cameras ===")
        try:
            # Print camera configurations
            print("Camera configurations:")
            for cam in config["cameras"]:
                print(f"  {cam}")

            # Initialize parallel cameras
            print("\nInitializing parallel cameras...")
            cameras = ParallelCameras(config["cameras"])
            
            # Test getting intrinsics for each camera
            print("\nTesting intrinsics:")
            for name in cameras.name_to_camera_args.keys():
                intrinsics = cameras.get_intrinsics(name)
                print(f"Camera {name} intrinsics:")
                print(f"  Matrix:\n{intrinsics['matrix']}")
                print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")

            # Test getting frames
            print("\nTesting frame capture:")
            for i in range(5):  # Get 5 frames
                print(f"\nCapturing frame set {i+1}")
                frames = cameras.get_frames()
                for camera_name, camera_frames in frames.items():
                    print(f"  Camera {camera_name}:")
                    for key, frame in camera_frames.items():
                        print(f"    {key} shape: {frame.shape}")
                        if key == "image":
                            # Optional: Save a test frame
                            if i == 0:  # Save first frame only
                                cv2.imwrite(f"test_frame_{camera_name}.jpg", frame)
                                print(f"    Saved test frame for {camera_name}")

        except Exception as e:
            print(f"\nError in parallel camera test: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            print("\nCleaning up parallel cameras...")
            del cameras

    # Run tests
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sequential', 'parallel', 'both'], 
                       default='both', help='Test mode')
    args = parser.parse_args()

    if args.mode in ['sequential', 'both']:
        test_sequential()
    if args.mode in ['parallel', 'both']:
        test_parallel()