#include <iostream>
#include <thread>
#include "feature_tracker.h"

std::queue<std::pair<double, cv::Mat>> img_buffer;

void readImage() {
	// read image from rosbag and put it in img_buffer
}

int main() {
	Parameters parameters;
	parameters.ROW = 480;
	parameters.COL = 752;
	parameters.FOCAL_LENGTH_X = 458.654;
	parameters.FOCAL_LENGTH_Y = 457.296;
	parameters.CENTER_X = 367.215;
	parameters.CENTER_Y = 248.375;

	parameters.MAX_FEATURE_NUM = 150;
	parameters.MIN_FEATURE_DIST = 30;

	parameters.SHOW_TRACK = 1;
	
	FeatureTracker feature_tracker(parameters);

	std::thread t1(readImage);

	while (1) {
		double time;
		cv::Mat image;

		//lock;
		if (!img_buffer.empty()) {
			time = img_buffer.front().first;
			image = img_buffer.front().second;
			img_buffer.pop();
		}
		//unlock;

		if (!image.empty()) {
			feature_tracker.trackFeatures(image, time);
		}

		cv::Mat trackImage = feature_tracker.getTrackImage();
		//opencv image publish

		//sleep 2ms
	}

	t1.join();
}

/*
1. FeaturePoint class -> struct
2. liftProjection, distortion -> FeatureTracker 안으로 합병

-----------------

- readImage() 함수 작성 rosbag c++ api
- mutex lock 사용방법
- opencv draw features
- outlier rejection
- prediction
- flowback
- publishing hz설정 어디서 하는지

-----------------

- distortion method / liftprojection algorithm
- outlier rejection의 fundamental matrix's use

*/