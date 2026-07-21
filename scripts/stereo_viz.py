#!/usr/bin/env python3
"""
Stereo camera visualization for Gazebo drone simulation.
Shows: left/right cameras, disparity, depth map, stereo distances, odometry.

Usage:
  python3 stereo_viz.py                    # display only
  python3 stereo_viz.py --record out.mp4   # record to MP4
  python3 stereo_viz.py --record out.mp4 --duration 30  # record 30 seconds
"""

import sys
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
import time
import math


class StereoViz(Node):
    def __init__(self):
        super().__init__("stereo_viz")

        self.left_img = None
        self.right_img = None
        self.depth_map = None
        self.stereo_dists = None
        self.odom = None
        self.left_info = None
        self.right_info = None

        self.sub_left = self.create_subscription(Image, "/left/image_raw", self._left_cb, 10)
        self.sub_right = self.create_subscription(Image, "/right/image_raw", self._right_cb, 10)
        self.sub_depth = self.create_subscription(Image, "/navigation_node/depth_map", self._depth_cb, 10)
        self.sub_dist = self.create_subscription(Float32MultiArray, "/navigation_node/stereo_distances", self._dist_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.sub_left_info = self.create_subscription(CameraInfo, "/left/camera_info", self._left_info_cb, 10)
        self.sub_right_info = self.create_subscription(CameraInfo, "/right/camera_info", self._right_info_cb, 10)

        self.bm = None
        self.frame_count = 0
        self.fps_time = time.time()
        self.fps = 0.0

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
        self.bm = cv2.StereoBM.create(128, 15)
        self.bm.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)
        self.bm.setPreFilterSize(9)
        self.bm.setPreFilterCap(31)
        self.bm.setMinDisparity(0)
        self.bm.setTextureThreshold(10)
        self.bm.setUniquenessRatio(15)
        self.bm.setSpeckleRange(32)
        self.bm.setSpeckleWindowSize(100)

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
        self.put_text(canvas, "FPS: %.1f" % self.fps, (200, 25), 0.6, (0, 255, 0))

        if self.odom:
            pos = self.odom.pose.pose.position
            q = self.odom.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            vel = self.odom.twist.twist.linear
            self.put_text(canvas, "POS: (%.1f, %.1f, %.1f)" % (pos.x, pos.y, pos.z), (350, 25), 0.5, (255, 255, 0))
            self.put_text(canvas, "YAW: %.1f deg" % math.degrees(yaw), (620, 25), 0.5, (255, 255, 0))
            self.put_text(canvas, "VEL: (%.2f, %.2f, %.2f)" % (vel.x, vel.y, vel.z), (780, 25), 0.5, (200, 200, 200))

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

        left_gray = cv2.cvtColor(self.left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(self.right_img, cv2.COLOR_RGB2GRAY)

        disp = self.compute_disparity(left_gray, right_gray)
        disp_color = self.colorize_disparity(disp)
        self.put_text(disp_color, "DISPARITY", (10, 25), 0.6, (255, 255, 255))

        depth = self.disparity_to_depth(disp)
        depth_color = self.colorize_depth(depth) if depth is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        self.put_text(depth_color, "DEPTH (JET)", (10, 25), 0.6, (255, 255, 255))

        left_viz = self.left_img.copy()
        right_viz = self.right_img.copy()
        self.put_text(left_viz, "LEFT CAM 640x480", (10, 25), 0.6, (0, 255, 0))
        self.put_text(right_viz, "RIGHT CAM 640x480", (10, 25), 0.6, (0, 255, 0))

        disp_resized = cv2.resize(disp_color, (640, 480))
        depth_resized = cv2.resize(depth_color, (640, 480))

        top_row = np.hstack([left_viz, right_viz])
        bottom_row = np.hstack([disp_resized, depth_resized])
        canvas = np.vstack([top_row, bottom_row])

        self.draw_hud(canvas)

        if self.stereo_dists and len(self.stereo_dists) >= 5:
            sidebar = np.zeros((canvas.shape[0], 200, 3), dtype=np.uint8)
            self.draw_distances(sidebar, 50, self.stereo_dists[:5])
            canvas = np.hstack([canvas, sidebar])

        return canvas

    def run(self, record_path=None, duration=None):
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

        print("Camera data OK. Starting visualization...")

        writer = None
        if record_path:
            # Build first frame to get dimensions
            frame = self.build_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(record_path, fourcc, 15.0, (w, h))
                print("Recording to %s (%dx%d @ 15fps)" % (record_path, w, h))
            else:
                print("ERROR: Could not build first frame!")
                return

        start_time = time.time()
        written = 0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            if self.left_img is None or self.right_img is None:
                continue

            self.frame_count += 1
            now = time.time()
            if now - self.fps_time >= 1.0:
                self.fps = self.frame_count / (now - self.fps_time)
                self.frame_count = 0
                self.fps_time = now

            frame = self.build_frame()
            if frame is None:
                continue

            if writer:
                writer.write(frame)
                written += 1
                elapsed = now - start_time
                if written % 30 == 0:
                    print("  Recorded %d frames (%.1fs elapsed)" % (written, elapsed))
                if duration and elapsed >= duration:
                    print("Duration limit reached (%.1fs)" % duration)
                    break
            else:
                cv2.imshow("Stereo Pipeline Visualization", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    fname = "/tmp/stereo_screenshot_%d.png" % int(time.time())
                    cv2.imwrite(fname, frame)
                    print("Saved: %s" % fname)

        if writer:
            writer.release()
            print("Video saved: %s (%d frames, %.1fs)" % (record_path, written, written / 15.0))

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Stereo visualization")
    parser.add_argument("--record", type=str, default=None, help="Record to MP4 file")
    parser.add_argument("--duration", type=float, default=None, help="Recording duration in seconds")
    args = parser.parse_args()

    rclpy.init()
    node = StereoViz()
    try:
        node.run(record_path=args.record, duration=args.duration)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Error: %s" % e)
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
