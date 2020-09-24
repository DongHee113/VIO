#include <iostream>
#include <thread>
#include "feature_tracker.h"

FeatureTracker feature_tracker;

void imgCallback() {
	//lock
	//put img in the buffer;
	//unlock
}

void tracking() {
	// ros img to auir img
	std::pair<int, Eigen::Matrix<double, 7, 1>> features;
	features = feature_tracker.trackFeatures(img, time);
	if (SHOW_TRACK) {
		cv::Mat image_track = feature_tracker.getTrackImage();
		pubTrackImage(image_track);
	}
}

int main() {


	std::thread feature_tracking_thread(tracking);
}