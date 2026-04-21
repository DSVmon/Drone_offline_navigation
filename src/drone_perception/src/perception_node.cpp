#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "sensor_msgs/msg/range.hpp"

class PerceptionNode : public rclcpp::Node
{
public:
  PerceptionNode()
  : Node("perception_node")
  {
    this->declare_parameter("min_safe_distance", 0.5);
    min_safe_distance_ = this->get_parameter("min_safe_distance").as_double();

    RCLCPP_INFO(this->get_logger(), "Drone Perception Node (Dual dToF Mode) started.");

    // Publishers for front and back alerts
    front_alert_pub_ = this->create_publisher<std_msgs::msg::Bool>("~/obstacle_front", 10);
    back_alert_pub_ = this->create_publisher<std_msgs::msg::Bool>("~/obstacle_back", 10);

    // Subscribers
    front_dtof_sub_ = this->create_subscription<sensor_msgs::msg::Range>(
      "/sensor/dtof_range", 10, std::bind(&PerceptionNode::frontCallback, this, std::placeholders::_1));
    
    back_dtof_sub_ = this->create_subscription<sensor_msgs::msg::Range>(
      "/sensor/dtof_back_range", 10, std::bind(&PerceptionNode::backCallback, this, std::placeholders::_1));
  }

private:
  void frontCallback(const sensor_msgs::msg::Range::SharedPtr msg)
  {
    auto alert = std_msgs::msg::Bool();
    alert.data = (msg->range < min_safe_distance_);
    if (alert.data) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Front obstacle at %.2f m", msg->range);
    }
    front_alert_pub_->publish(alert);
  }

  void backCallback(const sensor_msgs::msg::Range::SharedPtr msg)
  {
    auto alert = std_msgs::msg::Bool();
    alert.data = (msg->range < min_safe_distance_);
    if (alert.data) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Back obstacle at %.2f m", msg->range);
    }
    back_alert_pub_->publish(alert);
  }

  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr front_dtof_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr back_dtof_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr front_alert_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr back_alert_pub_;
  double min_safe_distance_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
