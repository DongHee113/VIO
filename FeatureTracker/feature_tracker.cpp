#include "feature_tracker.h"

int id_num = 0;

Eigen::Vector3d liftProjective(cv::Point2f &p) {
	Eigen::Vector3d v;

	double mx_d = p.x / FOCAL_LENGTH_X - CENTER_X;
	double my_d = p.y / FOCAL_LENGTH_Y - CENTER_Y;

	v << mx_d, my_d, 1.0;

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
			f[j++] = f[i];
		}
	}
	v.resize(j);
}

FeatureTracker::FeatureTracker() {
	//Read intrinsic parameter
}

cv::Mat getTrackImage() {

}

std::pair<int, Eigen::Matrix<double, 7, 1>> FeatureTracker::trackFeatures(aiur::sensor::data::camera::Image img, aiur::base::time::Millisecond time) {
	cv::Mat image = img;//Aiur2CV(img);

	readImage(image, time);

	if (feature_points_.curr_features.size() > 0) {
		tracking();
	}

	detecting();

	prev_img_ = curr_img_;
	feature_points_.prev_features = feature_points_.curr_features;
	curr_img_ = next_img_;
	feature_points_.curr_features = feature_points_.next_features;

	undistortPoints();

	calcPointVelocity();
}

void FeatureTracker::readImage(cv::Mat& img, double time) {
	curr_time_ = time;

	if (/*not initialized*/) {
		prev_img_ = curr_img_ = next_img_ = img;
	}
	else {
		next_img_ = img;
	}
}

void FeatureTracker::tracking() {
	feature_points_.next_features.clear();

	std::vector<uchar> status;
	std::vector<float> error;

	cv::calcOpticalFlowPyrLK(curr_img_, next_img_, feature_points_.curr_features, feature_points_.next_features, status, error, cv::Size(21, 21), 3);
	/*
		for (int i = 0; i < int(forw_pts.size()); i++)
				if (status[i] && !inBorder(forw_pts[i]))
					status[i] = 0; */

	reduceFeatures(feature_points_, status);

	for (auto& temp_track_cnt : feature_points_.track_cnt) {
		temp_track_cnt++;
	}
}

void FeatureTracker::detecting() {
	if (feature_points_.next_features.size() >= 8) {
		outlierRejection();
	}

	setMask();

	int num_of_features_to_track = MAX_FEATURE_NUM - static_cast<int>(feature_points_.next_features.size());
	if (num_of_features_to_track > 0)
	{
		cv::goodFeaturesToTrack(next_img_, new_features_, num_of_features_to_track, 0.01, MIN_FEATURE_DIST, mask_);
	}
	else {
		new_features_.clear();
	}

	addNewFeatures();
}

void FeatureTracker::outlierRejection() {
	if (forw_pts.size() >= 8)
	{
		ROS_DEBUG("FM ransac begins");
		TicToc t_f;
		vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
		for (unsigned int i = 0; i < cur_pts.size(); i++)
		{
			Eigen::Vector3d tmp_p;
			m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
			tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
			tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
			un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

			m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
			tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
			tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
			un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
		}

		vector<uchar> status;
		cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
		int size_a = cur_pts.size();
		reduceVector(prev_pts, status);
		reduceVector(cur_pts, status);
		reduceVector(forw_pts, status);
		reduceVector(cur_un_pts, status);
		reduceVector(ids, status);
		reduceVector(track_cnt, status);
		ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
		ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
	}
}

void FeatureTracker::setMask() {
	mask_ = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

	std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

	for (unsigned int i = 0; i < forw_pts.size(); i++)
		cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

	sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>>& a, const pair<int, pair<cv::Point2f, int>>& b)
		{
			return a.first > b.first;
		});

	forw_pts.clear();
	ids.clear();
	track_cnt.clear();

	for (auto& it : cnt_pts_id)
	{
		if (mask.at<uchar>(it.second.first) == 255)
		{
			forw_pts.push_back(it.second.first);
			ids.push_back(it.second.second);
			track_cnt.push_back(it.first);
			cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
		}
	}
}

void FeatureTracker::addNewFeatures() {
	for (auto& p : new_features_)
	{
		feature_points_.next_features.push_back(p);
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