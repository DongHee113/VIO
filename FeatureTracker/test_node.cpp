#include "feature_tracker.h"

int main() {
    FeatureTracker feature_tracker;

    feature_tracker.trackFeatures(img, time);

    auto features = feature_tracker.getFeatures();
    auto undistorted_features = feature_tracker.getUndistortedFeatures();
    auto feature_velocitys = feature_tracker.getFeatureVelocitys();
    auto feature_ids = feature_tracker.getFeatureIds();
}