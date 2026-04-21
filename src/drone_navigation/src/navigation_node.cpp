#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

class NavigationNode : public rclcpp::Node
{
public:
  NavigationNode()
  : Node("navigation_node")
  {
    // Declare parameters
    this->declare_parameter("image_topic", "/camera/image_raw");
    this->declare_parameter("imu_topic", "/imu/data");

    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string imu_topic = this->get_parameter("imu_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "Drone Navigation Node (VIO Wrapper) started.");
    RCLCPP_INFO(this->get_logger(), "Subscribing to Image: %s and IMU: %s",
                image_topic.c_str(), imu_topic.c_str());

    // Publisher for Odometry (Relative to node name /navigation)
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("~/odom", 10);

    // Subscribers for synchronized data
    image_sub_.subscribe(this, image_topic);
    imu_sub_.subscribe(this, imu_topic);

    // VIO typically needs tightly coupled IMU + Image
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), image_sub_, imu_sub_);

    sync_->registerCallback(std::bind(&NavigationNode::vioCallback, this,
                            std::placeholders::_1, std::placeholders::_2));
  }

private:
  void vioCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
    const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg)
  {
    (void)image_msg;
    (void)imu_msg;

    // Placeholder for VIO Estimation Logic

    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = this->now();
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";

    // Fill with placeholder position
    odom_msg.pose.pose.position.x = 0.0;
    odom_msg.pose.pose.position.y = 0.0;
    odom_msg.pose.pose.position.z = 1.0;

    odom_pub_->publish(odom_msg);
  }

  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Imu> SyncPolicy;

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NavigationNode>());
  rclcpp::shutdown();
  return 0;
}
