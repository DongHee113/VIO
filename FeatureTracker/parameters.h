#pragma once

struct Parameters {
	int ROW;
	int COL;

	double FOCAL_LENGTH_X;
	double FOCAL_LENGTH_Y;

	double CENTER_X;
	double CENTER_Y;

	double DISTORTION[4];

	int MAX_FEATURE_NUM;
	int MIN_FEATURE_DIST;

	double THRESHOLD_FOR_FUNDAMENTAL_MATRIX;

	bool SHOW_TRACK;
};