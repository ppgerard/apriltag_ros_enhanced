#pragma once

#include <apriltag/apriltag.h>
#include <functional>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>  // Add this for cv::Point3f


typedef std::function<geometry_msgs::msg::Transform(apriltag_detection_t* const, const std::array<double, 4>&, const double&)> pose_estimation_f;

// Add platform pose estimation function type
typedef std::function<geometry_msgs::msg::Transform(const std::vector<apriltag_detection_t*>&, const std::array<double, 4>&, const std::unordered_map<int, cv::Point3f>&, const std::unordered_map<int, double>&, double)> platform_pose_estimation_f;

extern const std::unordered_map<std::string, pose_estimation_f> pose_estimation_methods;
extern const platform_pose_estimation_f platform_pnp;
