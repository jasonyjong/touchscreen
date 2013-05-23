#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>

using namespace cv;

IplImage* GetThresholdedImage(IplImage* img, CvScalar minHSV, CvScalar maxHSV) {
	// Convert img to HSV space
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
	cvInRangeS(imgHSV,minHSV, maxHSV, imgThreshed);
	cvReleaseImage(&imgHSV);
	return imgThreshed;
}

void colorTracking() {
	CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
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
	int hue = 20;
	int saturation = 100;
	int value = 100;
	  cvCreateTrackbar("Hue","thresh", &hue, 245, NULL);
	//  cvCreateTrackbar("Saturation","thresh", &saturation, 245, NULL);
	//  cvCreateTrackbar("Value","thresh", &value, 245, NULL);


	// Show the image captured from the camera in the window and repeat
	while ( true ) {
		// Get one frame
		IplImage* frame = cvQueryFrame( capture );
		if ( !frame ) {
		  fprintf( stderr, "ERROR: Could not retrieve frame.\n" );
		  getchar();
		  break;
		}

		if(imgScribble == NULL) {
			imgScribble = cvCreateImage(cvGetSize(frame), 8, 3);
		}

		// Green Christmas Ornament
		CvScalar minHSV = cvScalar(81, 100, 100);
		CvScalar maxHSV = cvScalar(91, 255, 255);


//		    printf("Hue: %d Saturation: %d Value: %d \n", hue, saturation, value);
//		    CvScalar minHSV = cvScalar(hue, saturation, value);
//		    CvScalar maxHSV = cvScalar(hue+10, saturation+155, value+155);
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


		printf("Position (%d,%d)\n", posX, posY);

		// Draw a line only if positions are valid
		if (lastX > 0 && lastY > 0 && posX > 0 && posY > 0) {
		//    	printf("Draw line from (%d, %d) to (%d,%d)\n", posX, posY, lastX, lastY);
			cvLine(imgScribble, cvPoint(posX, posY), cvPoint(lastX, lastY), cvScalar(0, 255, 255), 5);
		}

		// Combine the scribbling to the frame
		cvAdd(frame, imgScribble, frame);
		//    cvShowImage("scribble", imgScribble);

		cvShowImage("thresh", imgThresh);
		cvShowImage( "video", frame );
		cvReleaseImage(&imgThresh);
		delete moments;

		// Wait for a keypress
		int c = cvWaitKey(10);
		if ( c != -1 ) break;
	}
	// Release the capture device housekeeping
	cvReleaseCapture( &capture );
	cvDestroyWindow( "mywindow" );

}

void circleTracking() {
	VideoCapture cap( CV_CAP_ANY );
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
		printf("There are %d circles!\n", circles.size());

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

/*int main( int argc, char** argv )
{
//	colorTracking();
	circleTracking();


  return 0;
}*/
