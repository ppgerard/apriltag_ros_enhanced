#include "pose_estimation.hpp"
#include <Eigen/Geometry>
#include <apriltag/apriltag_pose.h>
#include <apriltag/common/homography.h>
#include <opencv2/calib3d.hpp>
#include <tf2/convert.h>


geometry_msgs::msg::Transform
homography(apriltag_detection_t* const detection, const std::array<double, 4>& intr, double tagsize)
{
    apriltag_detection_info_t info = {detection, tagsize, intr[0], intr[1], intr[2], intr[3]};

    apriltag_pose_t pose;
    estimate_pose_for_tag_homography(&info, &pose);

    // rotate frame such that z points in the opposite direction towards the camera
    for(int i = 0; i < 3; i++) {
        // swap x and y axes
        std::swap(MATD_EL(pose.R, 0, i), MATD_EL(pose.R, 1, i));
        // invert z axis
        MATD_EL(pose.R, 2, i) *= -1;
    }

    return tf2::toMsg<apriltag_pose_t, geometry_msgs::msg::Transform>(const_cast<const apriltag_pose_t&>(pose));
}

geometry_msgs::msg::Transform
pnp(apriltag_detection_t* const detection, const std::array<double, 4>& intr, double tagsize)
{
    const std::vector<cv::Point3d> objectPoints{
        {-tagsize / 2, -tagsize / 2, 0},
        {+tagsize / 2, -tagsize / 2, 0},
        {+tagsize / 2, +tagsize / 2, 0},
        {-tagsize / 2, +tagsize / 2, 0},
    };

    const std::vector<cv::Point2d> imagePoints{
        {detection->p[0][0], detection->p[0][1]},
        {detection->p[1][0], detection->p[1][1]},
        {detection->p[2][0], detection->p[2][1]},
        {detection->p[3][0], detection->p[3][1]},
    };

    cv::Matx33d cameraMatrix;
    cameraMatrix(0, 0) = intr[0];// fx
    cameraMatrix(1, 1) = intr[1];// fy
    cameraMatrix(0, 2) = intr[2];// cx
    cameraMatrix(1, 2) = intr[3];// cy

    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, {}, rvec, tvec);

    return tf2::toMsg<std::pair<cv::Mat_<double>, cv::Mat_<double>>, geometry_msgs::msg::Transform>(std::make_pair(tvec, rvec));
}

// Add platform pose estimation using multiple tags
geometry_msgs::msg::Transform
platform_pnp_estimation(const std::vector<apriltag_detection_t*>& detections, 
                        const std::array<double, 4>& intr,
                        const std::unordered_map<int, cv::Point3f>& tag_positions,
                        const std::unordered_map<int, double>& tag_sizes,
                        double default_size)
{
    geometry_msgs::msg::Transform empty_transform;
    
    if(detections.size() < 1) {
        return empty_transform;
    }
    
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    
    // For each detected tag, add its 4 corners
    for(const auto& det : detections) {
        if(tag_positions.find(det->id) == tag_positions.end()) {
            continue;
        }
        
        const cv::Point3f& tag_center = tag_positions.at(det->id);
        
        // Use tag size from main tag configuration, fallback to default
        double tag_size = default_size;
        if(tag_sizes.count(det->id)) {
            tag_size = tag_sizes.at(det->id);
        }
        
        const double half_size = tag_size / 2.0;
        
        // Tag corners in platform coordinate system
        object_points.push_back(cv::Point3f(tag_center.x - half_size, tag_center.y - half_size, tag_center.z));
        object_points.push_back(cv::Point3f(tag_center.x + half_size, tag_center.y - half_size, tag_center.z));
        object_points.push_back(cv::Point3f(tag_center.x + half_size, tag_center.y + half_size, tag_center.z));
        object_points.push_back(cv::Point3f(tag_center.x - half_size, tag_center.y + half_size, tag_center.z));
        
        // Detected corners in image
        for(int i = 0; i < 4; i++) {
            image_points.push_back(cv::Point2f(det->p[i][0], det->p[i][1]));
        }
    }
    
    if(object_points.size() < 4) {
        return empty_transform;
    }
    
    // Camera matrix
    cv::Matx33d cameraMatrix;
    cameraMatrix(0, 0) = intr[0]; // fx
    cameraMatrix(1, 1) = intr[1]; // fy
    cameraMatrix(0, 2) = intr[2]; // cx
    cameraMatrix(1, 2) = intr[3]; // cy
    
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(object_points, image_points, cameraMatrix, {}, rvec, tvec);
    
    if(success) {
        return tf2::toMsg<std::pair<cv::Mat_<double>, cv::Mat_<double>>, geometry_msgs::msg::Transform>(std::make_pair(tvec, rvec));
    }
    
    return empty_transform;
}

const platform_pose_estimation_f platform_pnp = platform_pnp_estimation;

const std::unordered_map<std::string, pose_estimation_f> pose_estimation_methods{
    {"homography", homography},
    {"pnp", pnp},
};
