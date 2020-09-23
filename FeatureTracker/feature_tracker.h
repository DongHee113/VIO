#include <vector>

class FeaturePoints {
public:
    std::vector<int> ids;
    std::vector<int> track_cnt;
    
    std::vector<cv::Point2f> prev_features;
    std::vector<cv::Point2f> curr_features;
    std::vector<cv::Point2f> next_features;

    std::vector<cv::Point2f> undistorted_prev_features;
    std::vector<cv::Point2f> undistorted_curr_features;

    std::vector<cv::Point2f> velocity;
};

class FeatureTracker {
public:
    FeatureTracker();
    ~FeatureTracker();

    void trackFeatures(aiur::sensor::data::camera::Image img, aiur::base::time::Millisecond time);
private:
    void readImage();
    void tracking();
    void detecting();
    void outlierRejection();
    void setMask();
    void addNewFeatures();
    void undistortPoints();
    void calcPointVelocity();

    cv::Mat mask_;

    cv::Mat prev_img_;
    cv::Mat curr_img_;
    cv::Mat next_img_;

    FeaturePoints feature_points_;
    std::vector<cv::Point2> new_features_;

    double prev_time_;
    double curr_time_;
};