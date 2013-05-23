/*
 * blob_track.cpp
 *
 *  Created on: May 21, 2013
 *      Author: jjong
 */
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cvblobs/BlobResult.h>
#include <cvblobs/blob.h>

int main() {

	CBlobResult blobs;
	//CBlob *currentBlob;
	IplImage *original, *originalThr, *displayedImage;

	// load an image and threshold it
	original = cvLoadImage("pic1.png", 0);
	cvThreshold( original, originalThr, 100, 255, CV_THRESH_BINARY );

	// find non-white blobs in thresholded image
	blobs = CBlobResult( originalThr, NULL, 255);
	// exclude the ones smaller than param2 value
	/*blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS, 0 );

	// get mean gray color of biggest blob
	CBlob biggestBlob;
	CBlobGetMean getMeanColor( original );
	double meanGray;

	blobs.GetNthBlob( CBlobGetArea(), 0, biggestBlob );
	meanGray = getMeanColor( biggestBlob );

	// display filtered blobs
	cvMerge( originalThr, originalThr, originalThr, NULL, displayedImage );

	for (int i = 0; i < blobs.GetNumBlobs(); i++ )
	{
	        currentBlob = blobs.GetBlob(i);
	        currentBlob->FillBlob( displayedImage, CV_RGB(255,0,0));
	}*/
}
