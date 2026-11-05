import threading
import time
from typing import Tuple
import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag
from scipy.spatial.transform import Rotation


class AprilTagDetector:
    """
    RealSense AprilTag detector running in a background thread.

    The pose of the target tag is expressed in the camera frame used during
    training (see the `d435_camera_frame` site in the MuJoCo model).

    use case:
        detector = AprilTagDetector(tag_size=0.05, families="tag16h5", tag_id=0)
        detector.start()

        obs = detector.get_object_obs()          # [pos(3), orientation(...)]
        visible = detector.is_target_detected()

        detector.stop()
    """
    
    def __init__(self, 
                 tag_size: float = 50 / 1000,
                 families: str = "tag16h5",
                 tag_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 coordinate_transform: bool = True,
                 object_obs_size: int = 6
                 ):
        """
        Args:
            tag_size: AprilTag tag size
            families: AprilTag family
            tag_id: Target AprilTag ID
            resolution: Camera resolution
            fps: Camera frame rate
            coordinate_transform: Whether to apply coordinate transformation
            object_obs_size: Total size of object observation (position + orientation)
                           - 6: [pos(3), projected_gravity(3)]
                           - 7: [pos(3), quat(4)]
                           - 9: [pos(3), rot6d(6)]
                           - 12: [pos(3), rotmat(9)]
        """
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.resolution = resolution
        self.fps = fps
        self.coordinate_transform = coordinate_transform
        self.object_obs_size = object_obs_size
        
        # RealSense pipeline
        self.pipeline = None
        self.config = None
        self.camera_params = None
        
        # AprilTag detector
        options = apriltag.DetectorOptions(families=families)
        self.detector = apriltag.Detector(options)
        
        # Threading
        self.running = False
        self.thread = None
        self.detection_lock = threading.Lock()

        # Store only the latest pose instead of a buffer
        self.latest_pose = np.zeros(object_obs_size)
        self.last_object_quat = np.zeros(4)
        
        # Track detection status
        self.target_detected = False
        self.last_detection_time = 0.0

        # transform
        self.transform = np.eye(4,4)
        self.transform[:3, :3] = np.array([
                [0,  0,  1],   
                [-1,  0,  0],   
                [0, -1,  0]    
            ])

        self.flip = np.eye(4,4)
        self.flip[:3, :3] = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])
    
    def _initialize_camera(self):
        """Initialize RealSense"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        width, height = self.resolution
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        
        self.pipeline.start(self.config)
        
        # get intrinsic
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        
        self.camera_params = [
            intrinsics.fx,  # focal length x
            intrinsics.fy,  # focal length y
            intrinsics.ppx, # principal point x
            intrinsics.ppy  # principal point y
        ]
        
        print(f"Camera initialized: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
              f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    
    def _process_detection(self, detection) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single detection and return a tuple of (pose, quaternion)
        
        Returns:
            Tuple where:
                - first element is the pose array with size=object_obs_size
                - second element is the quaternion as [w, x, y, z]
        """
        # Get pose from detection
        pose, _, _ = self.detector.detection_pose(
            detection=detection,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )
        
        # Apply coordinate transformation
        pose = self.transform @ pose @ self.flip

        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        r = Rotation.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()  # [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
        
        # Determine orientation format based on object_obs_size
        orientation_size = self.object_obs_size - 3  # Subtract position (3D)
        
        if orientation_size == 3:
            # 6D total: [pos(3), projected_gravity(3)]
            gravity_world = np.array([0, 0, -1])
            orientation = rotation_matrix.T @ gravity_world  # Transform to tag frame
        elif orientation_size == 4:
            # 7D total: [pos(3), quat(4)]
            orientation = quat_wxyz
        elif orientation_size == 6:
            # 9D total: [pos(3), rot6d(6)]
            col1 = rotation_matrix[:, 0]  # First column (3,)
            col2 = rotation_matrix[:, 1]  # Second column (3,)
            orientation = np.concatenate([col1, col2], axis=-1)
        elif orientation_size == 9:
            # 12D total: [pos(3), rotmat(9)]
            orientation = rotation_matrix.flatten()  # Row-major flattening
        else:
            # Default to quaternion
            orientation = quat_wxyz

        # Combine position and orientation into pose vector
        pose_obs = np.concatenate([position, orientation])
        
        return pose_obs, quat_wxyz
    
    
    def _detection_loop(self):
        """Main detection loop running in background thread"""
        while self.running:
            try:
                # Get frame from camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to images
                color_image = np.asanyarray(color_frame.get_data())
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                
                # Detect AprilTags
                detections = self.detector.detect(gray_image)
                
                # Look for target tag
                target_detected = False
                for detection in detections:
                    if detection.tag_id == self.tag_id:
                        try:
                            pose_obs, object_quat = self._process_detection(detection)
                            
                            # Store latest pose and detection status (thread-safe)
                            with self.detection_lock:
                                self.latest_pose = pose_obs
                                self.last_object_quat = object_quat
                                self.target_detected = True
                                self.last_detection_time = time.time()
                            
                            target_detected = True
                            break  # Found target, no need to check other detections
                            
                        except Exception as e:
                            print(f"Error processing tag {detection.tag_id}: {e}")
                            continue
                
                # If target tag not detected this frame, update detection status
                if not target_detected:
                    with self.detection_lock:
                        self.target_detected = False
                       
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.01)  # Avoid busy waiting 
    
    def start(self):
        """Start the detector"""
        if self.running:
            print("Detector is already running")
            return
        
        self._initialize_camera()
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print("AprilTag detector started")
    
    def stop(self):
        """Stop the detector"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.pipeline:
            self.pipeline.stop()
        
        print("AprilTag detector stopped")
    
    def get_object_obs(self) -> np.ndarray:
        """
        Get latest object observations
        
        Returns:
            Array of latest pose with size=object_obs_size
            Format depends on object_obs_size:
            - 6: [pos(3), projected_gravity(3)]
            - 7: [pos(3), quat(4)]
            - 9: [pos(3), rot6d(6)]
            - 12: [pos(3), rotmat(9)]
        """
        with self.detection_lock:
            return self.latest_pose.copy()
    
    def get_last_quat(self) -> np.ndarray:
        """
        Get latest object quaternion [w, x, y, z]
        
        Returns:
            Quaternion array [w, x, y, z]
        """
        with self.detection_lock:
            return self.last_object_quat.copy()
    
    def is_target_detected(self) -> bool:
        """
        Check if target tag is currently detected
        
        Returns:
            True if target is detected in the latest frame, False otherwise
        """
        with self.detection_lock:
            return self.target_detected
    
    def get_time_since_last_detection(self) -> float:
        """
        Get time elapsed since last successful detection
        
        Returns:
            Time in seconds since last detection (0.0 if never detected)
        """
        with self.detection_lock:
            if self.last_detection_time == 0.0:
                return float('inf')  # Never detected
            return time.time() - self.last_detection_time
    
    def is_running(self) -> bool:
        """Check if detector is running"""
        return self.running
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()



if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from scripts.config import Config

    parser = argparse.ArgumentParser(description="Print AprilTag detections from the RealSense camera")
    parser.add_argument("--config", type=str, default="deploy/configs/g1_29dof_walk.yaml",
                        help="Config file whose `camera` block is used")
    parser.add_argument("--object_obs_size", type=int, default=7,
                        help="Object observation size to print (6, 7, 9 or 12)")
    args = parser.parse_args()

    detector = AprilTagDetector(object_obs_size=args.object_obs_size, **Config(args.config).camera)
    try:
        detector.start()
        time.sleep(2)
        while True:
            print(f"Target detected: {detector.is_target_detected()}")
            print(f"Time since last detection: {detector.get_time_since_last_detection():.3f}s")
            print(f"Obs: {detector.get_object_obs()}")
            print(f"Quat [w,x,y,z]: {detector.get_last_quat()}")
            print("---")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
