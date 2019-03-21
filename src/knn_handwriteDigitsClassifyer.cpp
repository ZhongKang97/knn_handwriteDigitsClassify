#include"../include/knn_handwriteDigitsClassifyer.h"
HD_Classify_KNN::HD_Classify_KNN()
{
	model = cv::ml::KNearest::create();
	trainPercent = 0.7;
	K = 5;
}

HD_Classify_KNN::~HD_Classify_KNN()
{
}
void HD_Classify_KNN::loadTrainDataFromImg(string &filename)
{
	cv::Mat img = cv::imread(filename);
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	int block = 20;
	int lenth_row = gray.rows / block;
	int lenth_col = gray.cols / block;
	cv::Mat data, labels;
	for (size_t i = 0; i < lenth_col; i++)
	{
		//列上的偏移量
		int offsetCol = i*block;
		for (size_t j = 0; j < lenth_row; j++)
		{
			int offsetRow = j*block;
			cv::Mat blockImg;
			gray(cv::Range(offsetRow, offsetRow + block), cv::Range(offsetCol, offsetCol + block)).copyTo(blockImg);
			data.push_back(blockImg.reshape(0, 1)); //展开成一行进行序列化
			labels.push_back((int)j / 5);
		}
	}
	data.convertTo(data, CV_32F);
	data.copyTo(train_test_data);
	labels.copyTo(train_test_labels);
	int samplesNum = data.rows;
	int trainNum = samplesNum*trainPercent;//取70%用于训练
	Mat trainData, trainLables;
	trainData = data(Range(0, trainNum), Range::all());
	trainLables = labels(Range(0, trainNum), Range::all());
	ptrainData = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLables);
}
void HD_Classify_KNN::trainAndTestKNN()
{
	//train
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(ptrainData);
	//test
	double train_ac = 0, test_ac = 0;
	for (size_t i = 0; i < train_test_data.rows; i++)
	{
		Mat sample = train_test_data.row(i);
		float r = model->predict(sample);
		r = std::abs(r - train_test_labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;
		if (i < train_test_data.rows*trainPercent)
			train_ac += r;
		else
			test_ac += r;
	}
	train_ac /= (train_test_data.rows*trainPercent);
	test_ac /= (train_test_data.rows*(1 - trainPercent));
	cout << "accuracy: train = " << train_ac * 100 << "%" << " test = " << test_ac * 100 << "%" << endl;
	model->save("kmean_digits.xml");
}
int HD_Classify_KNN::predict(cv::Mat inputDigit)
{
	Mat gray;
	cvtColor(inputDigit, gray, COLOR_BGR2GRAY);
	resize(gray, gray, Size(20, 20));
	inRange(gray, 125, 255, gray);
	imshow("binaryImg", gray);
	//equalizeHist(gray, gray);
	gray.convertTo(gray, CV_32F);
	Mat sample=gray.reshape(0, 1);
	float result = model->predict(sample);
	return (int)result;
	
}
void HD_Classify_KNN::getTrainedKNN(string & filename)
{
	model=cv::Algorithm::load<ml::KNearest>(filename);
}