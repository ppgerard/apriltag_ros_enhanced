// ros
#include "pose_estimation.hpp"
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>

// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions& options);

    ~AprilTagNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    // Change from single family to vector of families
    std::vector<apriltag_family_t*> tag_families;
    std::vector<std::function<void(apriltag_family_t*)>> tf_destructors;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<bool> profile;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    const image_transport::CameraSubscriber sub_cam;
    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    tf2_ros::TransformBroadcaster tf_broadcaster;

    pose_estimation_f estimate_pose = nullptr;

    // Add platform pose estimation members
    bool estimate_platform_pose;
    std::string platform_frame_id;
    std::unordered_map<int, cv::Point3f> platform_tag_positions;
    std::vector<int64_t> platform_tag_ids;

    void onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    // topics
    sub_cam(image_transport::create_camera_subscription(
        this,
        this->get_node_topics_interface()->resolve_topic_name("image_rect"),
        std::bind(&AprilTagNode::onCamera, this, std::placeholders::_1, std::placeholders::_2),
        declare_parameter("image_transport", "raw", descr({}, true)),
        rmw_qos_profile_sensor_data)),
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    tf_broadcaster(this)
{
    // read-only parameters
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size", true));

    // get tag names, IDs, families and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto tag_families_list = declare_parameter("tag.families", std::vector<std::string>{}, descr("tag families per id", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

    // Validate that all vectors have the same size
    if(!tag_families_list.empty() && ids.size() != tag_families_list.size()) {
        throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and families (" + std::to_string(tag_families_list.size()) + ") mismatch!");
    }

    // get method for estimating tag pose
    const std::string& pose_estimation_method =
        declare_parameter("pose_estimation_method", "pnp",
                          descr("pose estimation method: \"pnp\" (more accurate) or \"homography\" (faster), "
                                "set to \"\" (empty) to disable pose estimation",
                                true));

    if(!pose_estimation_method.empty()) {
        if(pose_estimation_methods.count(pose_estimation_method)) {
            estimate_pose = pose_estimation_methods.at(pose_estimation_method);
        }
        else {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown pose estimation method '" << pose_estimation_method << "'.");
        }
    }

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
    }

    // Collect unique families from tag specifications
    std::set<std::string> unique_families;
    for(const std::string& family : tag_families_list) {
        unique_families.insert(family);
    }

    // If no tag-specific families are specified, use default
    if(unique_families.empty()) {
        unique_families.insert("36h11");  // default family
    }

    // Initialize families
    for(const std::string& family_name : unique_families) {
        if(tag_fun.count(family_name)) {
            apriltag_family_t* tf = tag_fun.at(family_name).first();
            tag_families.push_back(tf);
            tf_destructors.push_back(tag_fun.at(family_name).second);
            apriltag_detector_add_family(td, tf);
            RCLCPP_INFO(get_logger(), "Added tag family: %s", family_name.c_str());
        }
        else {
            RCLCPP_ERROR(get_logger(), "Unsupported tag family: %s", family_name.c_str());
        }
    }
    
    if(tag_families.empty()) {
        throw std::runtime_error("No valid tag families specified!");
    }

    // Platform pose estimation parameters
    estimate_platform_pose = declare_parameter("platform.enable", false, descr("estimate platform pose using multiple tags"));
    platform_frame_id = declare_parameter("platform.frame_id", "platform", descr("platform frame id"));

    // Platform tag configuration
    platform_tag_ids = declare_parameter("platform.tag_ids", std::vector<int64_t>{}, descr("tag IDs on platform"));
    const auto platform_x = declare_parameter("platform.tag_x", std::vector<double>{}, descr("tag X positions on platform"));
    const auto platform_y = declare_parameter("platform.tag_y", std::vector<double>{}, descr("tag Y positions on platform"));
    const auto platform_z = declare_parameter("platform.tag_z", std::vector<double>{}, descr("tag Z positions on platform"));
    
    // Build platform tag positions map
    if(platform_tag_ids.size() == platform_x.size() && 
       platform_tag_ids.size() == platform_y.size() && 
       platform_tag_ids.size() == platform_z.size()) {
        for(size_t i = 0; i < platform_tag_ids.size(); i++) {
            platform_tag_positions[platform_tag_ids[i]] = cv::Point3f(platform_x[i], platform_y[i], platform_z[i]);
        }
    }
}

AprilTagNode::~AprilTagNode()
{
    apriltag_detector_destroy(td);
    // Destroy all families
    for(size_t i = 0; i < tag_families.size(); i++) {
        tf_destructors[i](tag_families[i]);
    }
}

void AprilTagNode::onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
                            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {msg_ci->p[0], msg_ci->p[5], msg_ci->p[2], msg_ci->p[6]};

    // check for valid intrinsics
    const bool calibrated = msg_ci->width && msg_ci->height &&
                            intrinsics[0] && intrinsics[1] && intrinsics[2] && intrinsics[3];

    if(estimate_pose != nullptr && !calibrated) {
        RCLCPP_WARN_STREAM(get_logger(), "The camera is not calibrated! Set 'pose_estimation_method' to \"\" (empty) to disable pose estimation and this warning.");
    }

    // convert to 8bit monochrome image
    const cv::Mat img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    // Add info message about total detections
    RCLCPP_INFO(get_logger(), "Detected %d AprilTags in current frame", zarray_size(detections));

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        if(estimate_pose != nullptr && calibrated) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = msg_img->header;
            // set child frame name by generic tag name or configured tag name
            tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
            const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
            tf.transform = estimate_pose(det, intrinsics, size);
            tfs.push_back(tf);
        }
    }

    // Platform pose estimation
    std::vector<apriltag_detection_t*> platform_detections;
    
    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);
        
        // Collect platform tags
        if(estimate_platform_pose && platform_tag_positions.count(det->id)) {
            platform_detections.push_back(det);
        }
    }
    
    // Estimate platform pose
    if(estimate_platform_pose && platform_detections.size() >= 1 && calibrated) {
        // Add info message about platform tags
        RCLCPP_INFO(get_logger(), "Platform pose estimation: Using %zu/%d detected tags for platform estimation", 
                   platform_detections.size(), zarray_size(detections));
                   
        geometry_msgs::msg::Transform platform_transform = platform_pnp(
            platform_detections, 
            intrinsics, 
            platform_tag_positions,
            tag_sizes,
            tag_edge_size);
        
        if(platform_transform.translation.x != 0 || platform_transform.translation.y != 0 || platform_transform.translation.z != 0) {
            geometry_msgs::msg::TransformStamped platform_tf;
            platform_tf.header = msg_img->header;
            platform_tf.child_frame_id = platform_frame_id;
            platform_tf.transform = platform_transform;
            tfs.push_back(platform_tf);
        }
    }

    pub_detections->publish(msg_detections);

    if(estimate_pose != nullptr)
        tf_broadcaster.sendTransform(tfs);

    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}
