#include"../include/knn_handwriteDigitsClassifyer.h"
int main(int argc, char** argv)
{
	HD_Classify_KNN classifer_knn;
	//string data = "D:/CODEing/OpenCV_codeSources/knn_handwriteDigitsClassify/data/cv_sample_png/digits.png";
	//classifer_knn.loadTrainDataFromImg(data);
	//classifer_knn.trainAndTestKNN();
	string knndata = "D:/CODEing/OpenCV_codeSources/knn_handwriteDigitsClassify/saveKNN/knn_digits.xml";
	classifer_knn.getTrainedKNN(knndata);
	Mat test = imread("D:/CODEing/OpenCV_codeSources/knn_handwriteDigitsClassify/data/cv_sample_png/8.jpg");
	imshow("testImage_8", test);
	cout << "the predict result is :" << classifer_knn.predict(test) << endl;
	test = imread("D:/CODEing/OpenCV_codeSources/knn_handwriteDigitsClassify/data/cv_sample_png/4.jpg");
	imshow("testImage_4", test);
	cout << "the predict result is :" << classifer_knn.predict(test) << endl;
	test = imread("D:/CODEing/OpenCV_codeSources/knn_handwriteDigitsClassify/data/cv_sample_png/5.jpg");
	imshow("testImage_5", test);
	cout << "the predict result is :" << classifer_knn.predict(test) << endl;
	waitKey(0);
	return 0;
}