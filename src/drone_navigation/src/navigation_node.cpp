#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/stereo_camera_model.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

class NavigationNode : public rclcpp::Node
{
public:
  NavigationNode()
  : Node("navigation_node")
  {
    // ... (existing parameters)
    
    // Create OpenCV window for automatic visualization (skip in headless)
    try {
      cv::namedWindow("Drone View: Rectified Left", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("Drone View: Disparity", cv::WINDOW_AUTOSIZE);
      cv::startWindowThread();
      headless_ = false;
    } catch (...) {
      RCLCPP_WARN(this->get_logger(), "Cannot create OpenCV windows (headless mode)");
      headless_ = true;
    }

    // Initialize BM (Block Matching) - Much faster than SGBM for real-time performance
    int num_disparities = 64; 
    int block_size = 21; // BM usually needs larger block size than SGBM (e.g., 15-21)
    
    bm_ = cv::StereoBM::create(num_disparities, block_size);
    bm_->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    bm_->setPreFilterSize(9);
    bm_->setPreFilterCap(31);
    bm_->setMinDisparity(0);
    bm_->setTextureThreshold(10);
    bm_->setUniquenessRatio(15);
    bm_->setSpeckleWindowSize(100);
    bm_->setSpeckleRange(32);
    bm_->setDisp12MaxDiff(1);
    // Declare parameters
    this->declare_parameter("left_image_topic", "/left/image_raw");
    this->declare_parameter("right_image_topic", "/right/image_raw");
    this->declare_parameter("left_info_topic", "/left/camera_info");
    this->declare_parameter("right_info_topic", "/right/camera_info");
    this->declare_parameter("imu_topic", "/imu/data");

    std::string left_topic = this->get_parameter("left_image_topic").as_string();
    std::string right_topic = this->get_parameter("right_image_topic").as_string();
    std::string left_info_topic = this->get_parameter("left_info_topic").as_string();
    std::string right_info_topic = this->get_parameter("right_info_topic").as_string();
    std::string imu_topic = this->get_parameter("imu_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "Subscribing to Left: %s, Right: %s", left_topic.c_str(), right_topic.c_str());

    RCLCPP_INFO(this->get_logger(), "Drone Navigation Node (Stereo VIO Wrapper) started.");

    // Publisher for Stereo-based multi-zone distance data
    stereo_dist_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("~/stereo_distances", 10);

    // Publisher for depth map (MAVRL architecture: 256x256 float32)
    depth_map_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/depth_map", 10);

    // Odom subscriber for calibration
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_x_ = msg->pose.pose.position.x;
      });

    // Debug publisher for rectified image
    rectified_pub_ = this->create_publisher<sensor_msgs::msg::Image>("~/debug_rectified_left", 10);

    // Camera Info Subscribers (using simple subscribers as they are usually static)
    left_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      left_info_topic, 10, [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        left_info_ = msg;
        checkCameraInfo();
      });
    right_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      right_info_topic, 10, [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        right_info_ = msg;
        checkCameraInfo();
      });

    // Subscribers for synchronized data
    left_sub_.subscribe(this, left_topic);
    right_sub_.subscribe(this, right_topic);
    imu_sub_.subscribe(this, imu_topic);

    // Sync Policy for Stereo + IMU (Increased back to 10 to prevent synchronization loss)
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), left_sub_, right_sub_, imu_sub_);

    sync_->registerCallback(std::bind(&NavigationNode::vioCallback, this,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }

private:
  void checkCameraInfo()
  {
    if (left_info_ && right_info_ && !model_initialized_) {
      stereo_model_.fromCameraInfo(left_info_, right_info_);
      model_initialized_ = true;
      RCLCPP_INFO(this->get_logger(), "Stereo Camera Model initialized. Baseline: %.4f m", stereo_model_.baseline());
    }
  }

  void vioCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_msg,
    const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg)
  {
    if (!model_initialized_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for CameraInfo to initialize model...");
      return;
    }

    // Heartbeat log removed for conciseness
    
    // 0. Process IMU Orientation (Tilt Correction)
    tf2::Quaternion q(
      imu_msg->orientation.x,
      imu_msg->orientation.y,
      imu_msg->orientation.z,
      imu_msg->orientation.w
    );
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    try {
      // 1. Convert ROS images to OpenCV Mat
      cv::Mat left_raw = cv_bridge::toCvShare(left_msg, "bgr8")->image;
      cv::Mat right_raw = cv_bridge::toCvShare(right_msg, "bgr8")->image;

      cv::Mat left_mono, right_mono;
      cv::cvtColor(left_raw, left_mono, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_raw, right_mono, cv::COLOR_BGR2GRAY);

      // 2. Rectification (Align images)
      cv::Mat left_rect_full, right_rect_full;
      stereo_model_.left().rectifyImage(left_mono, left_rect_full);
      stereo_model_.right().rectifyImage(right_mono, right_rect_full);

      // 2.5 Downsample for speed (Half resolution = 4x faster)
      cv::Mat left_rect, right_rect;
      cv::pyrDown(left_rect_full, left_rect);
      cv::pyrDown(right_rect_full, right_rect);

      // 4. Automatic Window Display (skip in headless)
      if (!headless_) {
        cv::imshow("Drone View: Rectified Left", left_rect);
        cv::waitKey(1);
      }

      // 5. Compute Disparity Map
      cv::Mat disparity_16s, disparity_8u;
      
      bm_->compute(left_rect, right_rect, disparity_16s);

      // Normalize for better visualization
      cv::Mat disp_visible;
      disparity_16s.convertTo(disp_visible, CV_32F, 1.0 / 16.0); // Convert to actual pixel disparity
      
      // Scale disparity back to full-res for distance calculation
      disp_visible *= 2.0;

      double min_val, max_val;
      cv::minMaxLoc(disp_visible, &min_val, &max_val);
      
      // Stretch to 0-255 range for visualization
      disp_visible.convertTo(disparity_8u, CV_8U, 255.0 / (max_val - min_val), 
                             -min_val * 255.0 / (max_val - min_val));

      cv::applyColorMap(disparity_8u, disparity_8u, cv::COLORMAP_JET);

      // 6. Multi-Zone Distance Calculation (Left, Center, Right, Top, Bottom)
      int zone_width = 80; 
      int zone_height = 60;
      
      double fy = stereo_model_.left().fy() * 0.5;
      int pitch_offset = static_cast<int>(pitch * fy);
      
      int start_x = (left_rect.cols - (zone_width * 3)) / 2;
      int start_y = (left_rect.rows - zone_height) / 2 - pitch_offset;
      start_y = std::max(0, std::min(start_y, left_rect.rows - zone_height));

      // Define ROIs for 5 zones: Left, Center, Right, Top, Bottom
      std::vector<cv::Rect> rois;
      // 0: Left, 1: Center, 2: Right (Horizontal)
      rois.push_back(cv::Rect(start_x, start_y, zone_width, zone_height));
      rois.push_back(cv::Rect(start_x + zone_width, start_y, zone_width, zone_height));
      rois.push_back(cv::Rect(start_x + 2 * zone_width, start_y, zone_width, zone_height));
      // 3: Top (Above Center), 4: Bottom (Below Center)
      int top_y = std::max(0, start_y - zone_height);
      int bottom_y = std::min(left_rect.rows - zone_height, start_y + zone_height);
      rois.push_back(cv::Rect(start_x + zone_width, top_y, zone_width, zone_height));
      rois.push_back(cv::Rect(start_x + zone_width, bottom_y, zone_width, zone_height));

      std::vector<float> zone_min_dists(5, 10.0f);
      std::vector<int> zone_valid_points(5, 0);

      cv::Mat debug_rect = left_rect.clone();
      cv::cvtColor(debug_rect, debug_rect, cv::COLOR_GRAY2BGR);

      for (size_t zone = 0; zone < rois.size(); ++zone) {
        cv::Scalar color = (zone == 1) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0); // Green for center
        if (zone >= 3) color = cv::Scalar(0, 255, 255); // Yellow for Top/Bottom
        
        cv::rectangle(debug_rect, rois[zone], color, 2);

        for (int v = rois[zone].y; v < rois[zone].y + rois[zone].height; ++v) {
          for (int u = rois[zone].x; u < rois[zone].x + rois[zone].width; ++u) {
            float disp = disp_visible.at<float>(v, u);
            if (disp > 0.1f) {
              float z = stereo_model_.getZ(disp);
              if (z > 0.1f && z < 15.0f) {
                if (z < zone_min_dists[zone]) zone_min_dists[zone] = z;
                zone_valid_points[zone]++;
              }
            }
          }
        }
      }

      // Show windows with debug overlays (skip in headless)
      if (!headless_) {
        cv::imshow("Drone View: Rectified Left", debug_rect);
        
        cv::Mat disparity_with_roi = disparity_8u.clone();
        for (size_t zone = 0; zone < rois.size(); ++zone) {
          cv::Scalar color = (zone == 1) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
          if (zone >= 3) color = cv::Scalar(0, 255, 255);
          cv::rectangle(disparity_with_roi, rois[zone], color, 2);
        }
        cv::imshow("Drone View: Disparity", disparity_with_roi);
        cv::waitKey(1);
      }

      // Publish multi-zone distances
      auto dist_msg = std_msgs::msg::Float32MultiArray();
      dist_msg.data = zone_min_dists;
      stereo_dist_pub_->publish(dist_msg);

      // 7. Depth Map for MAVRL (disparity → depth in meters → uint8 [0,255])
      // Convert disparity to depth: Z = f * B / d
      cv::Mat depth_map_32f;
      disp_visible.copyTo(depth_map_32f);

      float focal_length = stereo_model_.left().fx() * 0.5; // half-res
      float baseline = stereo_model_.baseline();

      for (int v = 0; v < depth_map_32f.rows; ++v) {
        for (int u = 0; u < depth_map_32f.cols; ++u) {
          float disp = depth_map_32f.at<float>(v, u);
          if (disp > 0.1f) {
            float z = (focal_length * baseline) / disp;
            depth_map_32f.at<float>(v, u) = std::min(z, 12.0f); // clamp max 12m
          } else {
            depth_map_32f.at<float>(v, u) = 12.0f; // far = no data
          }
        }
      }

      // Normalize to [0, 255] uint8 (matching MAVRL format)
      cv::Mat depth_map_uint8;
      depth_map_32f.convertTo(depth_map_uint8, CV_8U, 255.0 / 12.0);

      // Center crop to square (matching MAVRL)
      int crop_size = std::min(depth_map_uint8.cols, depth_map_uint8.rows);
      int crop_x = (depth_map_uint8.cols - crop_size) / 2;
      int crop_y = (depth_map_uint8.rows - crop_size) / 2;
      cv::Mat depth_cropped = depth_map_uint8(cv::Rect(crop_x, crop_y, crop_size, crop_size));

      // Resize to 256x256 for MAVRL policy input
      cv::Mat depth_map_resized;
      cv::resize(depth_cropped, depth_map_resized, cv::Size(256, 256), 0, 0, cv::INTER_AREA);

      // Publish depth map as mono8 image (0-255, matching MAVRL)
      auto depth_header = left_msg->header;
      depth_header.frame_id = "camera_link";
      auto depth_img_msg = cv_bridge::CvImage(depth_header, "mono8", depth_map_resized).toImageMsg();
      depth_map_pub_->publish(*depth_img_msg);

      auto debug_img_msg = cv_bridge::CvImage(left_msg->header, "bgr8", debug_rect).toImageMsg();
      rectified_pub_->publish(*debug_img_msg);

    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
  }

  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Imu> SyncPolicy;

  message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;
  sensor_msgs::msg::CameraInfo::SharedPtr left_info_;
  sensor_msgs::msg::CameraInfo::SharedPtr right_info_;
  image_geometry::StereoCameraModel stereo_model_;
  cv::Ptr<cv::StereoBM> bm_;
  bool model_initialized_ = false;
  bool headless_ = false;
  double current_x_ = 0.0;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr stereo_dist_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectified_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NavigationNode>());
  rclcpp::shutdown();
  return 0;
}
