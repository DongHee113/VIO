#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

#include "parameters.h"

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

	std::pair<int, Eigen::Matrix<double, 7, 1>> trackFeatures(aiur::sensor::data::camera::Image img, aiur::base::time::Millisecond time);
	cv::Mat getTrackImage();
private:
	void readImage(cv::Mat &img, double time);
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
	std::vector<cv::Point2f> new_features_;

	std::map<int, cv::Point2f> undistorted_prev_features_map_;

	double prev_time_;
	double curr_time_;
};