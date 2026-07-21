#!/usr/bin/env python3
"""
Record stereo visualization to MP4 using imageio (H.264).
Must be run after Gazebo + navigation are started.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
import time
import math
import sys

import imageio.v3 as iio


class StereoRecorder(Node):
    def __init__(self):
        super().__init__("stereo_recorder")

        self.left_img = None
        self.right_img = None
        self.depth_map = None
        self.stereo_dists = None
        self.odom = None
        self.left_info = None
        self.right_info = None

        self.create_subscription(Image, "/left/image_raw", self._left_cb, 10)
        self.create_subscription(Image, "/right/image_raw", self._right_cb, 10)
        self.create_subscription(Image, "/navigation_node/depth_map", self._depth_cb, 10)
        self.create_subscription(Float32MultiArray, "/navigation_node/stereo_distances", self._dist_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(CameraInfo, "/left/camera_info", self._left_info_cb, 10)
        self.create_subscription(CameraInfo, "/right/camera_info", self._right_info_cb, 10)

        self.bm = None

    def _left_cb(self, msg):
        self.left_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def _right_cb(self, msg):
        self.right_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def _depth_cb(self, msg):
        self.depth_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)

    def _dist_cb(self, msg):
        self.stereo_dists = msg.data

    def _odom_cb(self, msg):
        self.odom = msg

    def _left_info_cb(self, msg):
        self.left_info = msg

    def _right_info_cb(self, msg):
        self.right_info = msg

    def init_stereo_bm(self):
        self.bm = cv2.StereoBM.create(64, 7)  # faster: 64 disp, block=7
        self.bm.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)
        self.bm.setPreFilterSize(5)
        self.bm.setPreFilterCap(31)
        self.bm.setMinDisparity(0)
        self.bm.setTextureThreshold(10)
        self.bm.setUniquenessRatio(10)
        self.bm.setSpeckleRange(16)
        self.bm.setSpeckleWindowSize(50)

    def compute_disparity(self, left_gray, right_gray):
        if self.bm is None:
            self.init_stereo_bm()
        return self.bm.compute(left_gray, right_gray)

    def disparity_to_depth(self, disp):
        if self.left_info is None:
            return None
        fx = self.left_info.k[0]
        B = abs(self.right_info.p[3]) / fx if self.right_info else 0.12
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 0
        depth[valid] = fx * B * 16.0 / disp[valid].astype(np.float32)
        return np.clip(depth, 0, 12.0)

    def colorize_depth(self, depth, max_val=6.0):
        norm = np.clip(depth / max_val, 0, 1)
        return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    def colorize_disparity(self, disp):
        valid = disp > 0
        disp_vis = np.zeros_like(disp, dtype=np.uint8)
        disp_vis[valid] = np.clip(disp[valid] / 16.0, 0, 128).astype(np.uint8)
        return cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)

    def put_text(self, img, text, pos=(10, 25), scale=0.6, color=(0, 255, 0)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    def draw_hud(self, canvas):
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        self.put_text(canvas, "STEREO VIZ", (10, 25), 0.7, (0, 255, 255))

        if self.odom:
            pos = self.odom.pose.pose.position
            q = self.odom.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            vel = self.odom.twist.twist.linear
            self.put_text(canvas, "POS: (%.1f, %.1f, %.1f)" % (pos.x, pos.y, pos.z), (350, 25), 0.5, (255, 255, 0))
            self.put_text(canvas, "YAW: %.1f deg" % math.degrees(yaw), (620, 25), 0.5, (255, 255, 0))

    def draw_distances(self, canvas, y_start, dists):
        labels = ["Left", "Center", "Right", "Ceiling", "Floor"]
        bar_w = 120
        bar_h = 18
        x_start = 10
        self.put_text(canvas, "STEREO DISTANCES:", (x_start, y_start + 18), 0.5, (255, 255, 255))
        for i, (label, d) in enumerate(zip(labels, dists)):
            y = y_start + 25 + i * (bar_h + 5)
            cv2.rectangle(canvas, (x_start, y), (x_start + bar_w, y + bar_h), (50, 50, 50), -1)
            fill = int(bar_w * min(d / 5.0, 1.0))
            color = (0, 255, 0) if d > 1.5 else (0, 255, 255) if d > 0.5 else (0, 0, 255)
            cv2.rectangle(canvas, (x_start, y), (x_start + fill, y + bar_h), color, -1)
            cv2.rectangle(canvas, (x_start, y), (x_start + bar_w, y + bar_h), (100, 100, 100), 1)
            self.put_text(canvas, "%s: %.1fm" % (label, d), (x_start + bar_w + 5, y + 14), 0.4, color)

    def build_frame(self):
        if self.left_img is None or self.right_img is None:
            return None

        # Use depth map from navigation_node (256x256 mono8)
        if self.depth_map is not None:
            depth_color = cv2.applyColorMap(self.depth_map, cv2.COLORMAP_JET)
        else:
            depth_color = np.zeros((256, 256, 3), dtype=np.uint8)
        self.put_text(depth_color, "DEPTH (nav 256x256)", (5, 18), 0.4, (255, 255, 255))

        # Cameras
        left_viz = cv2.resize(self.left_img, (560, 420))
        right_viz = cv2.resize(self.right_img, (560, 420))
        self.put_text(left_viz, "LEFT CAM", (5, 18), 0.5, (0, 255, 0))
        self.put_text(right_viz, "RIGHT CAM", (5, 18), 0.5, (0, 255, 0))

        depth_resized = cv2.resize(depth_color, (1120, 420))

        top_row = np.hstack([left_viz, right_viz])
        canvas = np.vstack([top_row, depth_resized])

        self.draw_hud(canvas)

        if self.stereo_dists and len(self.stereo_dists) >= 5:
            sidebar = np.zeros((canvas.shape[0], 160, 3), dtype=np.uint8)
            self.draw_distances(sidebar, 50, self.stereo_dists[:5])
            canvas = np.hstack([canvas, sidebar])

        return canvas

    def run(self, output_path, duration):
        self.init_stereo_bm()
        rclpy.spin_once(self, timeout_sec=0.1)

        print("Waiting for camera data...")
        timeout = time.time() + 15
        while time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.left_img is not None and self.right_img is not None:
                break

        if self.left_img is None:
            print("ERROR: No camera data received!")
            return

        print("Camera data OK. Building first frame...")
        frame = self.build_frame()
        if frame is None:
            print("ERROR: Could not build frame!")
            return

        h, w = frame.shape[:2]
        print("Recording: %s (%dx%d, %ds)" % (output_path, w, h, duration))

        # Use av directly for reliable H.264 recording
        import av
        container = av.open(output_path, mode="w")
        stream = container.add_stream("h264", rate=30)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"

        def encode_frame(bgr_frame):
            rgb_frame = av.VideoFrame.from_ndarray(bgr_frame[:, :, ::-1], format="rgb24")
            for packet in stream.encode(rgb_frame):
                container.mux(packet)

        encode_frame(frame)  # first frame

        start_time = time.time()
        written = 1

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            if self.left_img is None or self.right_img is None:
                continue

            now = time.time()
            elapsed = now - start_time
            if elapsed >= duration:
                break

            frame = self.build_frame()
            if frame is None:
                continue

            # Write frame
            encode_frame(frame)
            written += 1

            if written % 30 == 0:
                print("  %d frames (%.1fs / %ds)" % (written, elapsed, duration))

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print("Done! %d frames, %.1fs" % (written, written / 30.0))
        print("Saved: %s" % output_path)


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/Git_store/Drone_offline_navigation/stereo_visualization.mp4"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    rclpy.init()
    node = StereoRecorder()
    try:
        node.run(output_path, duration)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
