#include <opencv2/opencv.hpp>

using namespace cv;

void HarrisCorners(Mat& srcImage, Mat& dstImage, double alpha)
{
	//转化为灰度图
	Mat grayImage;
	if (srcImage.channels() == 3)
		cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	else
		grayImage = srcImage.clone();

	//分别计算x，y方向上的梯度
	Mat xKernel = (Mat_<double>(1, 3) << -1, 0, 1);
	Mat yKernel = xKernel.t();

	Mat Ix, Iy;
	filter2D(grayImage, Ix, CV_64F, xKernel);
	filter2D(grayImage, Iy, CV_64F, yKernel);

	Mat Ix2, Iy2, Ixy;
	Ix2 = Ix.mul(Ix);
	Iy2 = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);

	Mat gaussKernel = getGaussianKernel(7, 1);
	filter2D(Ix2, Ix2, CV_64F, gaussKernel);
	filter2D(Iy2, Iy2, CV_64F, gaussKernel);
	filter2D(Ixy, Ixy, CV_64F, gaussKernel);

	//计算角点响应值，越大代表越像是角点
	Mat cornerStrength(grayImage.size(), CV_64FC1);
	for (int i = 0; i < grayImage.rows; i++)
	{
		for (int j = 0; j < grayImage.cols; j++)
		{
			double det = Ix2.at<double>(i, j)*Iy2.at<double>(i, j) - Ixy.at<double>(i, j)*Ixy.at<double>(i, j);
			double trace = Ix2.at<double>(i, j) + Iy2.at<double>(i, j);
			cornerStrength.at<double>(i, j) = det - alpha*trace*trace;
		}
	}

	double maxStrength;
	minMaxLoc(cornerStrength,NULL, &maxStrength, NULL, NULL);
	Mat dilated, localMax;
	dilate(cornerStrength, dilated, Mat());                        //将角点响应图膨胀后输入dilated，Mat()表示用原图某像素3×3范围内的最大值代替dilated中该位置的值
	compare(cornerStrength, dilated, localMax, CMP_EQ);            //提取局部极大值并抑制局部非极大值。将cornerStrength和dilated每个元素比较，若相等就将localMax中相应位置的值设为255。CMP_EQ表示相等操作
	
	Mat cornerMap;
	double qualityLevel = 0.05;
	double thresh = qualityLevel*maxStrength;
	cornerMap = cornerStrength > thresh;                           //cornerStrength>thresh 会将cornerStrength中大于thresh的像素值置为255
	bitwise_and(cornerMap, localMax, cornerMap);                   //位运算“与”，求得既是局部极大值又大于某个阈值的像素点，cornerMap是二值图

	dstImage = cornerMap.clone();
}

void drawCorners(Mat& image, Mat& cornerMap)
{
	Mat_<uchar>::const_iterator it = cornerMap.begin<uchar>();
	Mat_<uchar>::const_iterator itend = cornerMap.end<uchar>();
	for (int i = 0; it != itend; i++, it++)
	{
		if (*it)
			circle(image, Point(i%image.cols, i / image.cols), 3, Scalar(0, 0, 255), 1);      //i%image.cols:计算非零点所在的列数，i/image.cols:计算非零点所在的行数
	}
}

int main()
{
	Mat srcImage = imread("C:\\Users\\LIUU\\Pictures\\gate.jpg",1);
	Mat dstImage;
	HarrisCorners(srcImage, dstImage, 0.01);
	drawCorners(srcImage, dstImage);
	imshow("aa", srcImage);
	waitKey(0);
}