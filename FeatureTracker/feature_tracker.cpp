#include "feature_tracker.h"

int id_num = 0;

Eigen::Vector3d liftProjective(cv::Point2f& p) {
	Eigen::Vector3d v;

	double mx_d = p.x / FOCAL_LENGTH_X - CENTER_X;
	double my_d = p.y / FOCAL_LENGTH_Y - CENTER_Y;

	if (1 /*no distortion*/) {
		v << mx_d, my_d, 1.0;
	}

	return v;
}
bool inBorder(const cv::Point2f& pt)
{
	const int BORDER_SIZE = 1;
	int img_x = cvRound(pt.x);
	int img_y = cvRound(pt.y);
	return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}
void reduceFeatures(FeaturePoints& f, std::vector<uchar> status)
{
	int j = 0;
	for (size_t i = 0; i < f.ids.size(); i++) {
		if (status[i]) {
			f.ids[j] = f.ids[i];
			f.track_cnt[j] = f.track_cnt[i];
			f.prev_features[j] = f.prev_features[i];
			f.curr_features[j] = f.curr_features[i];
			j++;
		}
	}

	f.ids.resize(j);
	f.track_cnt.resize(j);
	f.prev_features.resize(j);
	f.curr_features.resize(j);
}

FeatureTracker::FeatureTracker() {
	//Read intrinsic parameter
}

cv::Mat getTrackImage() {

}

std::pair<int, Eigen::Matrix<double, 7, 1>> FeatureTracker::trackFeatures(aiur::sensor::data::camera::Image img, aiur::base::time::Millisecond time) {
	curr_img_ = img;//Aiur2CV(img);
	curr_time_ = time;//Aiur2CV(time)
	

	if (feature_points_.prev_features.size() > 0) {
		tracking();
	}

	detecting();

	undistortPoints();

	calcPointVelocity();

	prev_img_ = curr_img_;
	prev_time_ = curr_time_;
	feature_points_.prev_features = feature_points_.curr_features;
}


void FeatureTracker::tracking() {
	feature_points_.curr_features.clear();

	std::vector<uchar> status;
	std::vector<float> error;

	cv::calcOpticalFlowPyrLK(prev_img_, curr_img_, feature_points_.prev_features, feature_points_.curr_features, status, error, cv::Size(21, 21), 3);
	
	for (size_t i = 0; i < feature_points_.curr_features.size(); i++) {
		if (status[i] && !inBorder(feature_points_.curr_features[i])) {
			status[i] = 0;
		}
	}

	reduceFeatures(feature_points_, status);

	for (auto& temp_track_cnt : feature_points_.track_cnt) {
		temp_track_cnt++;
	}
}

void FeatureTracker::detecting() {
	setMask();

	int num_of_features_to_track = MAX_FEATURE_NUM - static_cast<int>(feature_points_.curr_features.size());
	if (num_of_features_to_track > 0)
	{
		cv::goodFeaturesToTrack(curr_img_, new_features_, num_of_features_to_track, 0.01, MIN_FEATURE_DIST, mask_);
	}
	else {
		new_features_.clear();
	}

	addNewFeatures();
}

void FeatureTracker::setMask() {
	mask_ = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

	std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

	for (size_t i = 0; i < feature_points_.curr_features.size(); i++) {
		cnt_pts_id.push_back(std::make_pair(feature_points_.track_cnt[i], std::make_pair(feature_points_.curr_features[i], feature_points_.ids[i])));
	}

	sort(cnt_pts_id.begin(), cnt_pts_id.end(), 
		[](const std::pair<int, std::pair<cv::Point2f, int>>& a, const std::pair<int, std::pair<cv::Point2f, int>>& b)
		{
			return a.first > b.first;
		});

	feature_points_.curr_features.clear();
	feature_points_.ids.clear();
	feature_points_.track_cnt.clear();

	for (auto& it : cnt_pts_id)
	{
		if (mask_.at<uchar>(it.second.first) == 255)
		{
			feature_points_.curr_features.push_back(it.second.first);
			feature_points_.ids.push_back(it.second.second);
			feature_points_.track_cnt.push_back(it.first);
			cv::circle(mask_, it.second.first, MIN_FEATURE_DIST, 0, -1);
		}
	}
}

void FeatureTracker::addNewFeatures() {
	for (auto& p : new_features_)
	{
		feature_points_.curr_features.push_back(p);
		feature_points_.ids.push_back(id_num++);
		feature_points_.track_cnt.push_back(1);
	}
}

void FeatureTracker::undistortPoints() {
	feature_points_.undistorted_curr_features.clear();

	Eigen::Vector3d projected3d;

	for (size_t i = 0; i < feature_points_.curr_features.size(); i++) {
		projected3d = liftProjective(feature_points_.curr_features[i]);
		feature_points_.undistorted_curr_features.push_back(cv::Point2f(projected3d.x() / projected3d.z(), projected3d.y() / projected3d.z()));
	}
}

void FeatureTracker::calcPointVelocity() {
	if (!undistorted_prev_features_map_.empty()) {
		double dt = curr_time_ - prev_time_;
		feature_points_.velocity.clear();

		for (size_t i = 0; i < feature_points_.undistorted_curr_features.size(); i++) {
			std::map<int, cv::Point2f>::iterator it;
			it = undistorted_prev_features_map_.find(feature_points_.ids[i]);
			if (it != undistorted_prev_features_map_.end()) {
				double v_x = (feature_points_.undistorted_curr_features[i].x - it->second.x) / dt;
				double v_y = (feature_points_.undistorted_curr_features[i].y - it->second.y) / dt;

				feature_points_.velocity.push_back(cv::Point2f(v_x, v_y));
			}
			else {
				feature_points_.velocity.push_back(cv::Point2f(0, 0));
			}
		}
	}
	else {
		for (size_t i = 0; i < feature_points_.undistorted_curr_features.size(); i++) {
			feature_points_.velocity.push_back(cv::Point2f(0, 0));
		}
	}

	for (size_t i; i < feature_points_.ids.size(); i++) {
		undistorted_prev_features_map_.insert(std::make_pair(feature_points_.ids[i], feature_points_.undistorted_curr_features[i]));
	}
}