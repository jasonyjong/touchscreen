#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <math.h>
//#include <cvblobs/BlobResult.h>

using namespace cv;

bool Combine(cv::Rect r, cv::Rect r1, cv::Rect r2);

IplImage* GetThresholdedImage(IplImage* img, CvScalar minHSV, CvScalar maxHSV) {
	// Convert img to HSV space
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
	cvInRangeS(imgHSV,minHSV, maxHSV, imgThreshed);
	cvReleaseImage(&imgHSV);
	return imgThreshed;
}

int Area(cv::Rect r) {
	return abs(r.tl().x-r.br().x)*abs(r.tl().y-r.br().y);
}

void colorTracking() {
	CvCapture* capture = cvCaptureFromCAM( 0 );
	if ( !capture ) {
		fprintf( stderr, "ERROR: Could not initialize capturing. \n" );
		getchar();
	}
	// Create a window in which the captured images will be presented
	cvNamedWindow( "video", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "thresh", CV_WINDOW_AUTOSIZE);
	IplImage* imgScribble = NULL;
	int posX = 0;
	int posY = 0;
	int hue = 10;
	int saturation = 60;
	int value = 65;
	  cvCreateTrackbar("Hue","thresh", &hue, 245, NULL);
	  cvCreateTrackbar("Saturation","thresh", &saturation, 245, NULL);
	  cvCreateTrackbar("Value","thresh", &value, 245, NULL);


	// Show the image captured from the camera in the window and repeat
	while ( true ) {
		// Get one frame
		IplImage* frame = cvQueryFrame( capture );
		if ( !frame ) {
		  fprintf( stderr, "ERROR: Could not retrieve frame.\n" );
		  getchar();
		  break;
		}

//		if(imgScribble == NULL) {
//			imgScribble = cvCreateImage(cvGetSize(frame), 8, 3);
//		}

		// Green Christmas Ornament
//		CvScalar minHSV = cvScalar(30, 150, 150);
//		CvScalar maxHSV = cvScalar(65, 255, 255);
//		CvScalar minHSV = cvScalar(hue, saturation, value);
//		CvScalar maxHSV = cvScalar(hue+10, saturation+200, value+200);
		CvScalar minHSV = cvScalar(hue, saturation, value);
		CvScalar maxHSV = cvScalar(65, 255, 255);

	    printf("Hue: %d Saturation: %d Value: %d \n", hue, saturation, value);
		IplImage* imgThresh = GetThresholdedImage(frame, minHSV, maxHSV);

		// Calculate the moments to estimate the position of the ball
		CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
		cvMoments(imgThresh, moments, 1);
		double moment10 = cvGetSpatialMoment(moments, 1, 0);
		double moment01 = cvGetSpatialMoment(moments, 0, 1);
		double area = cvGetCentralMoment(moments, 0, 0);

		int lastX = posX;
		int lastY = posY;
		posX = moment10/area;
		posY = moment01/area;
//		printf("Area: %f\n", area);

		printf("Position (%d,%d)\n", posX, posY);

		Mat mImage = Mat(imgThresh);
		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;
		findContours( mImage,
		                  contours,
		                  hierarchy,
		                  CV_RETR_TREE,
		                  CV_CHAIN_APPROX_SIMPLE,
		                  Point(0, 0) );

		std::vector<std::vector<Point> > contours_poly( contours.size() );
		std::vector<cv::Rect> boundRect( contours.size() );
		for( int i = 0; i < contours.size(); i++ )
		{
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
			boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		}

		// Cluster the rectangles:
		std::vector<cv::Rect> clusterRect( contours.size() );
		for (int i=0; i < boundRect.size(); i++) {
			cv::Rect r = boundRect[i];
			for (int j=0; j<clusterRect.size(); j++) {
				cv::Rect rmod;
				if (Combine(rmod, r, clusterRect[j])) {
					printf("REMOVED\n");
					clusterRect.erase(clusterRect.begin() + j);
					j--;
					r = rmod;
				}
			}

			if (Area(r) > 3) {
				r.tl().x = max(0,r.tl().x-100);
				r.br().x = r.br().x+100;
				r.tl().y = max(0,r.tl().y-100);
				r.br().y = r.br().y+100;
				r = cv::Rect(cv::Point(max(0,r.tl().x-20),max(0,r.tl().y-20)),cv::Point(r.br().x+20,r.br().y+20));
				clusterRect.push_back(r);
			}
		}
		printf("COUNT++++++++++++++++++ %i\n", clusterRect.size());

		// Debug purposes: draw bonding rects
		Mat tmp = Mat::zeros( mImage.size(), CV_8UC3 );
		for( int i = 0; i< clusterRect.size(); i++ )
		  rectangle( tmp, clusterRect[i].tl(), clusterRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );

		IplImage a = tmp;
		IplImage* b = &a;
		cvShowImage("thresh", b);
		cvReleaseImage(&b);

		std::vector<cv::Point> rect_tl;
		std::vector<cv::Point> rect_br;
		for( int i = 0; i < clusterRect.size(); i++ )
		{
		    rect_tl.push_back(clusterRect[i].tl());
		    rect_br.push_back(clusterRect[i].br());
		}

		//cv::Mat drawing = cv::Mat::zeros( red_image.size(), CV_8UC3 );

		int count = 0;
		for( int i = 0; i < rect_tl.size(); i++ )
		{
			int x_distance = abs(rect_tl[i].x - rect_br[i].x);
			int y_distance = abs(rect_tl[i].y - rect_br[i].y);

			if ( (x_distance > 2) && (y_distance > 2) )
			{
				count++;
		    }
		}

		std::cout<<"Count: "<<count<<std::endl;

		// Draw a line only if positions are valid
		if (area > 100) {
			if (lastX > 0 && lastY > 0 && posX > 0 && posY > 0) {
			//    	printf("Draw line from (%d, %d) to (%d,%d)\n", posX, posY, lastX, lastY);
//				cvLine(imgScribble, cvPoint(posX, posY), cvPoint(lastX, lastY), cvScalar(0, 255, 255), 5);
				cvLine(frame, cvPoint(posX, posY), cvPoint(lastX, lastY), cvScalar(0, 255, 255), 5);
			}
		}
		else printf("Too noisy to track\n");

		// Combine the scribbling to the frame
//		cvAdd(frame, imgScribble, frame);
		//    cvShowImage("scribble", imgScribble);

		//cvShowImage("thresh", imgThresh);
		cvShowImage( "video", frame );
		//cvReleaseImage(&imgThresh);
		delete moments;

		// Wait for a keypress
		int c = cvWaitKey(10);
		if ( c != -1 ) break;
	}
	// Release the capture device housekeeping
	cvReleaseCapture( &capture );
	cvDestroyWindow( "mywindow" );

}

bool Combine(cv::Rect r, cv::Rect r1, cv::Rect r2) {

	int x1c = r1.tl().x;
	int x2c = r1.br().x;
	int y1c = r1.tl().y;
	int y2c = r1.br().y;

	int x1 = r2.tl().x;
	int x2 = r2.br().x;
	int y1 = r2.tl().y;
	int y2 = r2.br().y;

	bool a = (x1>=x1c && x1<=x2c) || (x2>=x1c && x2<=x2c);
	bool b = (y1>=y1c && y1<=y2c) || (y2>=y1c && y2<=y2c);
	bool c = (x1c>=x1 && x1c<=x2) || (x2c>=x1 && x2c<=x2);
	bool d = (y1c>=y1 && y1c<=y2) || (y2c>=y1 && y2c<=y2);
	if ((a&&b)||(c&&d)) {
		int xb = min(x1,x1c);
		int xt = max(x2,x2c);
		int yb = min(y1,y1c);
		int yt = max(y2,y2c);
		r = cv::Rect(cv::Point(xt,yt), cv::Point(xb,yb));
		return true;
	}

	return false;
}

void circleTracking() {
	VideoCapture cap( 0 );
	if ( !cap.isOpened() ) {
		fprintf( stderr, "ERROR: Could not initialize capturing. \n" );
		getchar();
	}
	// Create a window in which the captured images will be presented
	namedWindow( "video", CV_WINDOW_AUTOSIZE );

	// Show the image captured from the camera in the window and repeat
	while ( true ) {
		// Get one frame
		Mat frame, frame_gray, mask;
		cap >> frame;

//		// Convert to HSV
//		cvtColor(frame, mask, CV_BGR2HSV);
//		// Green Christmas Ornament
//		Scalar minHSV = Scalar(81, 100, 100);
//		Scalar maxHSV = Scalar(91, 255, 255);
//		inRange(mask, minHSV, maxHSV, frame_gray);
//		Mat element21 = getStructuringElement(MORPH_RECT, Size(21,21), Point(10,10));
//		morphologyEx(frame_gray, frame_gray, MORPH_OPEN, element21);
//		Mat element11 = getStructuringElement(MORPH_RECT, Size(11,11), Point(5,5));
//		morphologyEx(frame_gray, frame_gray, MORPH_CLOSE, element21);

		// Convert to grayscale
		cvtColor(frame, frame_gray, CV_BGR2GRAY);

		// Reduce noise
		GaussianBlur(frame_gray, frame_gray, Size(9,9), 2, 2);

		vector<Vec3f> circles;

		// Apply Hough Transform to find circles
		HoughCircles( frame_gray, circles, CV_HOUGH_GRADIENT, 1, frame_gray.rows/8, 200, 100, 0, 0 );
		// printf("There are %d circles!\n", circles.size());

		for (size_t i = 0; i < circles.size(); i++) {
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
			// circle outline
			circle( frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}
		imshow( "video", frame );
//		imshow( "gray", frame_gray );


		// Wait for a keypress
		int c = cvWaitKey(100);
		if ( c != -1 ) break;
	}
}



IplImage* trackColor(IplImage *image, CvPoint *position, int hue) {
	CvScalar minHSV = cvScalar(hue, 100, 100);
	CvScalar maxHSV = cvScalar(hue+10, 255, 255);

	IplImage* imgHSV = cvCreateImage(cvGetSize(image), 8, 3);
	cvCvtColor(image, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(image), 8, 1);
	cvInRangeS(imgHSV,minHSV, maxHSV, imgThreshed);
	cvReleaseImage(&imgHSV);

	// Calculate the moments to estimate the position of the ball
	CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
	cvMoments(imgThreshed, moments, 1);
	double moment10 = cvGetSpatialMoment(moments, 1, 0);
	double moment01 = cvGetSpatialMoment(moments, 0, 1);
	double area = cvGetCentralMoment(moments, 0, 0);

	if (area < 100) {
		printf("Too noisy to track, detected area: %f\n", area);
		position->x = -1;
		position->y = -1;
	}
	else {
		position->x = moment10/area;
		position->y = moment01/area;
	}
	delete moments;
	return imgThreshed;
}

double calculateBearing(CvPoint red, CvPoint blue) {
	double a = blue.x-red.x;
	double b = blue.y-red.y;
	if (blue.y > red.y) {
		if (a == 0) {
			return 90.0;
		}
		else {
			return atan(b/a) * (180/3.14159);
		}
	}
	else {
		if (a == 0) {
			return 270.0;
		}
		else {
			return atan(b/a) * (180/3.14159) + 180;
		}
	}
}

void blobTracking() {
	CvCapture* capture = cvCaptureFromCAM( 1 ); //CV_CAP_ANY
	if ( !capture ) {
		fprintf( stderr, "ERROR: Could not initialize capturing. \n" );
		getchar();
	}
	// Create a window in which the captured images will be presented
	cvNamedWindow( "video", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "thresh", CV_WINDOW_AUTOSIZE);


	// Show the image captured from the camera in the window and repeat
	while ( true ) {
		// Get one frame
		IplImage* frame = cvQueryFrame( capture );
		if ( !frame ) {
		  fprintf( stderr, "ERROR: Could not retrieve frame.\n" );
		  getchar();
		  break;
		}


		CvPoint yellowPos;
		CvPoint bluePos;
		CvPoint redPos;
		IplImage* yellowThresh= trackColor(frame, &yellowPos, 25);
		IplImage* blueThresh= trackColor(frame, &bluePos, 100);
		IplImage* redThresh= trackColor(frame, &redPos, 164);

		// Draw a line only if positions are valid
		if (yellowPos.x > 0 && yellowPos.y > 0 && bluePos.x > 0 && bluePos.y > 0 && redPos.x && redPos.y) {
			cvLine(frame, yellowPos, bluePos, cvScalar(0, 255, 255), 5);
			cvLine(frame, yellowPos, redPos, cvScalar(0, 255, 255), 5);
			cvLine(frame, redPos, bluePos, cvScalar(0, 255, 255), 5);

			double bearing = calculateBearing(redPos, bluePos);
			printf("Bearing: %f\n",bearing);

		}

		cvShowImage("thresh", yellowThresh);
		cvShowImage("thresh", blueThresh);
		cvShowImage("thresh", redThresh);
		cvShowImage( "video", frame );
		cvReleaseImage(&yellowThresh);
		cvReleaseImage(&blueThresh);
		cvReleaseImage(&redThresh);

		// Wait for a keypress
		int c = cvWaitKey(10);
		if ( c != -1 ) break;
	}
	// Release the capture device housekeeping
	cvReleaseCapture( &capture );
	cvDestroyWindow( "mywindow" );
}


int main( int argc, char** argv )
{
	colorTracking();
//	circleTracking();
//	blobTracking();


  return 0;
}
