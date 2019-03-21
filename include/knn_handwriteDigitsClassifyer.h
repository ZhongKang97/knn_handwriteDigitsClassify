#ifndef _KNN_HANDWRITECLASSIFYER_H
#define _KNN_HANDWRITECLASSIFYER_H
#include<iostream>
#include<opencv2/ml.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
using namespace std;
using namespace cv;
class HD_Classify_KNN
{
public:
	HD_Classify_KNN();
	~HD_Classify_KNN();
	void loadTrainDataFromImg(string &filename);
	void trainAndTestKNN();
	void getTrainedKNN(string &filename);
	int predict(cv::Mat inputDigits);
private:
	cv::Ptr<cv::ml::TrainData> ptrainData;
	cv::Ptr<cv::ml::KNearest> model;
	Mat train_test_data;
	Mat train_test_labels;
	float trainPercent;
	int K;
};
#endif
