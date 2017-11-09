#include <opencv2/opencv.hpp>

using namespace cv;

void HarrisCorners(Mat& srcImage, Mat& dstImage, double alpha)
{
	//ת��Ϊ�Ҷ�ͼ
	Mat grayImage;
	if (srcImage.channels() == 3)
		cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	else
		grayImage = srcImage.clone();

	//�ֱ����x��y�����ϵ��ݶ�
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

	//����ǵ���Ӧֵ��Խ�����Խ���ǽǵ�
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
	dilate(cornerStrength, dilated, Mat());                        //���ǵ���Ӧͼ���ͺ�����dilated��Mat()��ʾ��ԭͼĳ����3��3��Χ�ڵ����ֵ����dilated�и�λ�õ�ֵ
	compare(cornerStrength, dilated, localMax, CMP_EQ);            //��ȡ�ֲ�����ֵ�����ƾֲ��Ǽ���ֵ����cornerStrength��dilatedÿ��Ԫ�رȽϣ�����Ⱦͽ�localMax����Ӧλ�õ�ֵ��Ϊ255��CMP_EQ��ʾ��Ȳ���
	
	Mat cornerMap;
	double qualityLevel = 0.05;
	double thresh = qualityLevel*maxStrength;
	cornerMap = cornerStrength > thresh;                           //cornerStrength>thresh �ὫcornerStrength�д���thresh������ֵ��Ϊ255
	bitwise_and(cornerMap, localMax, cornerMap);                   //λ���㡰�롱����ü��Ǿֲ�����ֵ�ִ���ĳ����ֵ�����ص㣬cornerMap�Ƕ�ֵͼ

	dstImage = cornerMap.clone();
}

void drawCorners(Mat& image, Mat& cornerMap)
{
	Mat_<uchar>::const_iterator it = cornerMap.begin<uchar>();
	Mat_<uchar>::const_iterator itend = cornerMap.end<uchar>();
	for (int i = 0; it != itend; i++, it++)
	{
		if (*it)
			circle(image, Point(i%image.cols, i / image.cols), 3, Scalar(0, 0, 255), 1);      //i%image.cols:�����������ڵ�������i/image.cols:�����������ڵ�����
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