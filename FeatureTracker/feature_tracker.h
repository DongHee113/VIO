#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

#include "parameters.h"

typedef struct FeaturePoints {
public:
	std::vector<int> ids;
	std::vector<int> track_cnt;

	std::vector<cv::Point2f> prev_features;
	std::vector<cv::Point2f> curr_features;

	std::vector<cv::Point2f> undistorted_curr_features;

	std::vector<cv::Point2f> velocity;
}FeaturePoints;

class FeatureTracker {
public:
	FeatureTracker();
	FeatureTracker(Parameters &parameter);

	std::pair<int, Eigen::Matrix<double, 7, 1>> trackFeatures(cv::Mat &img, double &time);
	cv::Mat getTrackImage();
private:
	void tracking();
	void detecting();
	void setMask();
	void addNewFeatures();
	void undistortPoints();
	void calcPointVelocity();

	Eigen::Vector3d liftProjective(cv::Point2f& p);
	void distortion(Eigen::Vector2d p_u, Eigen::Vector2d& d_u);

	bool inBorder(const cv::Point2f& pt);

	const int ROW;
	const int COL;

	const double FOCAL_LENGTH_X;
	const double FOCAL_LENGTH_Y;

	const double CENTER_X;
	const double CENTER_Y;

	const double K1;
	const double K2;
	const double P1;
	const double P2;

	const int MAX_FEATURE_NUM;
	const int MIN_FEATURE_DIST;

	const bool SHOW_TRACK;

	cv::Mat mask_;

	cv::Mat prev_img_;
	cv::Mat curr_img_;

	FeaturePoints feature_points_;
	std::vector<cv::Point2f> new_features_;

	std::map<int, cv::Point2f> undistorted_prev_features_map_;

	double prev_time_;
	double curr_time_;
};