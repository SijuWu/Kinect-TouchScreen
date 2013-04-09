
#include "stdafx.h"
#include <stdio.h>
#include <iostream>

#include<XnCppWrapper.h>
#include <highgui.h>
#include <math.h>


#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>


using namespace std;
using namespace cv;

#define PI 3.14159265;

cv::Mat colorMouse;
cv::Mat depthMouse;

 IplImage*  imgMouse;

vector<Point>screenCorners;

vector<int>leftUpDepth;
vector<int>rightUpDepth;
vector<int>rightDownDepth;
vector<int>leftDownDepth;

bool screenCatch=false;
int frameCount=0;

vector<Point2f> corners;

int leftUpIndex;
int rightDownIndex;
int leftDownIndex;
int rightUpIndex;

vector<int>cornerDepth;
vector<cv::Point3f>cornerCoordinates;

void GetDesktopResolution(int& horizontal, int& vertical)
{
   RECT desktop;
   // Get a handle to the desktop window
   const HWND hDesktop = GetDesktopWindow();
   // Get the size of screen to the variable desktop
   GetWindowRect(hDesktop, &desktop);
   // The top left corner will have coordinates (0,0)
   // and the bottom right corner will have coordinates
   // (horizontal, vertical)
   horizontal = desktop.right;
   vertical = desktop.bottom;
}

void mouseEvent(int evt,int x,int y,int flags,void* param)
{
	if(evt==CV_EVENT_LBUTTONDOWN)
	{
		Point3_<uchar>* p = colorMouse.ptr<Point3_<uchar> >(y,x);
		/*cout<<x<<" "<<y<<" "<<(int)p->x<<" "<<(int)p->y<<" "<<(int)p->z<<"\n";*/
		cout<<x<<" "<<y<<" "<<cvGet2D(imgMouse,y,x).val[0]<<"\n";
	}
}

void checkCornerOrder(int &leftUpIndex,int &rightDownIndex,int &leftDownIndex,int &rightUpIndex)
{
	leftUpIndex=NULL;
	rightDownIndex=NULL;
	leftDownIndex=NULL;
	rightUpIndex=NULL;

	if(corners.size()==4)
	{
		//calculate the distance between (0,0) and the corner
		vector<float> dis;
		dis.push_back(sqrt(pow(corners[0].x,2)+pow(corners[0].y,2)));
		dis.push_back(sqrt(pow(corners[1].x,2)+pow(corners[1].y,2)));
		dis.push_back(sqrt(pow(corners[2].x,2)+pow(corners[2].y,2)));
		dis.push_back(sqrt(pow(corners[3].x,2)+pow(corners[3].y,2)));


		float disMin=10000;
		float disMax=0;

		//the left up corner is the nearest to (0,0), the right down corner is the farest to (0,0)
		for(int i=0;i<dis.size();i++)
		{
			if(dis[i]<disMin)
			{
				disMin=dis[i];
				leftUpIndex=i;
			}
			if(dis[i]>disMax)
			{
				disMax=dis[i];
				rightDownIndex=i;
			}
		}

		//Get the index of the left down corner and of the right up corner
		for(int i=0;i<dis.size();i++)
		{
			if(i!=leftUpIndex&&i!=rightDownIndex)
			{
				if(leftDownIndex==NULL)
					leftDownIndex=i;
				else
					rightUpIndex=i;
			}
		}
		//If the x coordinate of the right up corner is smaller than that of the left down corner, switch them
		if(corners[rightUpIndex].x<corners[leftDownIndex].x)
		{
			int changeIndex=leftDownIndex;
			leftDownIndex=rightUpIndex;
			rightUpIndex=changeIndex;
		}
	}
}

void CheckOpenNIError( XnStatus result, string status )  
{   
    if( result != XN_STATUS_OK )   
        cerr << status << " Error: " << xnGetStatusString( result ) << endl;  
}  

int main(int argc, char** argv)
{
	XnStatus result = XN_STATUS_OK;    
	xn::DepthMetaData depthMD;  
	xn::ImageMetaData imageMD;  

	//OpenCV  
	IplImage*  imgDepth16u=cvCreateImage(cvSize(640,480),IPL_DEPTH_16U,1);  
	IplImage*  depthShow=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1);  

	IplImage* imgRGB8u=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3);  
	IplImage* imageShow=cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3); 

	
    char key=0;  
  
    // context   
    xn::Context context;   
    result = context.Init();   
    CheckOpenNIError( result, "initialize context" );    
  
    // creategenerator    
    xn::DepthGenerator depthGenerator;    
    result = depthGenerator.Create( context );   
    CheckOpenNIError( result, "Create depth generator" );    
    xn::ImageGenerator imageGenerator;  
    result = imageGenerator.Create( context );   
    CheckOpenNIError( result, "Create image generator" );  
    
    //map mode    
    XnMapOutputMode mapMode;   
    mapMode.nXRes = 640;    
    mapMode.nYRes = 480;   
    mapMode.nFPS = 30;   
    result = depthGenerator.SetMapOutputMode( mapMode );    
    result = imageGenerator.SetMapOutputMode( mapMode );    
   
    // correct view port    
    depthGenerator.GetAlternativeViewPointCap().SetViewPoint( imageGenerator );   
   
    //read data  
    result = context.StartGeneratingAll();    
 
    result = context.WaitAndUpdateAll();    
  
    while( (key!=27) && !(result = context.WaitAndUpdateAll( ))  )   
    {    
        //get meta data  
        depthGenerator.GetMetaData(depthMD);   
        imageGenerator.GetMetaData(imageMD);  
  
        //OpenCV output  
		//Depth map
        memcpy(imgDepth16u->imageData,depthMD.Data(),640*480*2);  
        cvConvertScale(imgDepth16u,depthShow,255/4096.0,0); 
		imgMouse=imgDepth16u;

		//Color map
        memcpy(imgRGB8u->imageData,imageMD.Data(),640*480*3);  
        cvCvtColor(imgRGB8u,imageShow,CV_RGB2BGR);  

		//Save depth image to a matrix
		cv::Mat depth(depthShow,0);
		cv::Mat result=depth;
		cv::medianBlur(depth,result,5);

		cv::Mat depthNew;
		cv::cvtColor(result,depthNew,CV_GRAY2BGR);
		
		//Save color image to a matrix
		cv::Mat color(imageShow,0);
		
		cv::Mat colorGray;
		cv::cvtColor(color,colorGray,CV_BGR2GRAY);
		colorMouse=colorGray;

		cv::Mat fullScreen;
		cv::cvtColor(color,fullScreen,CV_BGR2GRAY);

		//Segmentation of the color gray map
		for(int i=0;i<colorGray.size().height;i++)
		{
			for(int j=0;j<colorGray.size().width;j++)
			{
				if(colorGray.at<uchar>(i,j)<100)
					colorGray.at<uchar>(i,j)=0;
				else
					colorGray.at<uchar>(i,j)=255;
				fullScreen.at<uchar>(i,j)=255;
			}
		}
		
		//Add mouse event
		cvSetMouseCallback("color", mouseEvent, 0);
		cvSetMouseCallback("depth",mouseEvent,0);
		
		///Find the contour of the screen
	    Mat src_gray;
		int thresh = 100;
		int max_thresh = 255;
		RNG rng(12345);

		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		//Smoothing of the image
		 blur( colorGray, src_gray, Size(3,3) );
		// Detect edges using canny
		Canny( src_gray, canny_output, thresh, thresh*2, 3 );
		// Find contours
		findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		/// Find the convex hull object for each contour
		vector<vector<Point> >hull( contours.size() );
		for( int i = 0; i < contours.size(); i++ )
		{  convexHull( Mat(contours[i]), hull[i], false ); }

		/// Draw contours + hull results
		Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
		int size=contours.size();
	
		//Find the convex hull with the largest area
		double maxArea=0;
		int hullIndex=0;
		for(int i=0;i<hull.size();i++)
		{
			if(contourArea(hull[i],true)>maxArea)
			{
				maxArea=contourArea(hull[i],true);
				hullIndex=i;
			}	
		}

		Scalar colorContour = Scalar( 255, 255, 255 );
		drawContours( drawing, hull, hullIndex, colorContour, 1, 8, vector<Vec4i>(), 0, Point() );

		cv::Mat drawingGray;
		cv::cvtColor(drawing,drawingGray,CV_BGR2GRAY);

		/// Detect four corners by Shi-Tomasi algorithm
		int maxCorners=4;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = false;
		double k = 0.04;

		/// Copy the image on which the convex hull is drawn
		Mat copy;
		copy = drawing.clone();

		/// Apply corner detection
		goodFeaturesToTrack( drawingGray,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(),
			blockSize,
			useHarrisDetector,
			k );

		/// Draw corners detected
		int r = 4;
		for( int i = 0; i < corners.size(); i++ )
		{
			circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255),
		rng.uniform(0,255)), -1, 8, 0 ); 
		}

		//When the key S is pressed, it begins to find the depth value of four corners
		if(key=='s')
		{
			//Find the index of four corners
			checkCornerOrder(leftUpIndex,rightDownIndex,leftDownIndex,rightUpIndex);
			
			//Save corners according to the clockwise order
			screenCorners.clear();
			screenCorners.push_back(corners[leftUpIndex]);
			screenCorners.push_back(corners[rightUpIndex]);
			screenCorners.push_back(corners[rightDownIndex]);
			screenCorners.push_back(corners[leftDownIndex]);
			screenCatch=true;
		}

		//If the detection is started, begin to find depth values and save them
		if(screenCatch==true)
		{
			if(frameCount<=20)
			{
				//Find index of four corners
				checkCornerOrder(leftUpIndex,rightDownIndex,leftDownIndex,rightUpIndex);

				//If the convex hull of the screen is not correct, skip to the next frame
				if(corners[leftUpIndex].x>screenCorners[0].x+10||corners[leftUpIndex].x<screenCorners[0].x-10
					||corners[leftUpIndex].y>screenCorners[0].y+10||corners[leftUpIndex].y<screenCorners[0].y-10)
				{
					continue;
				}
				if(corners[rightUpIndex].x>screenCorners[1].x+10||corners[rightUpIndex].x<screenCorners[1].x-10
					||corners[rightUpIndex].y>screenCorners[1].y+10||corners[rightUpIndex].y<screenCorners[1].y-10)
				{
					continue;
				}
				if(corners[rightDownIndex].x>screenCorners[2].x+10||corners[rightDownIndex].x<screenCorners[2].x-10
					||corners[rightDownIndex].y>screenCorners[2].y+10||corners[rightDownIndex].y<screenCorners[2].y-10)
				{
					continue;
				}
				if(corners[leftDownIndex].x>screenCorners[3].x+10||corners[leftDownIndex].x<screenCorners[3].x-10
					||corners[leftDownIndex].y>screenCorners[3].y+10||corners[leftDownIndex].y<screenCorners[3].y-10)
				{
					continue;
				}

				//Get the depth value of four corners
				int depthLeftUp=cvGet2D(imgDepth16u,corners[leftUpIndex].y,corners[leftUpIndex].x).val[0];
				int depthRightUp=cvGet2D(imgDepth16u,corners[rightUpIndex].y,corners[rightUpIndex].x).val[0];
				int depthRightDown=cvGet2D(imgDepth16u,corners[rightDownIndex].y,corners[rightDownIndex].x).val[0];
				int depthLeftDown=cvGet2D(imgDepth16u,corners[leftDownIndex].y,corners[leftDownIndex].x).val[0];

				//Save only depth values which are not 0
				if(depthLeftUp!=0 && depthRightUp!=0 && depthRightDown!=0 && depthLeftDown!=0)
				{
					leftUpDepth.push_back(cvGet2D(imgDepth16u,corners[leftUpIndex].y,corners[leftUpIndex].x).val[0]);
					rightUpDepth.push_back(cvGet2D(imgDepth16u,corners[rightUpIndex].y,corners[rightUpIndex].x).val[0]);
					rightDownDepth.push_back(cvGet2D(imgDepth16u,corners[rightDownIndex].y,corners[rightDownIndex].x).val[0]);
					leftDownDepth.push_back(cvGet2D(imgDepth16u,corners[leftDownIndex].y,corners[leftDownIndex].x).val[0]);

					cout<<cvGet2D(imgDepth16u,corners[leftUpIndex].y,corners[leftUpIndex].x).val[0]<<" "
						<<cvGet2D(imgDepth16u,corners[rightUpIndex].y,corners[rightUpIndex].x).val[0]<<" "
						<<cvGet2D(imgDepth16u,corners[rightDownIndex].y,corners[rightDownIndex].x).val[0]<<" "
						<<cvGet2D(imgDepth16u,corners[leftDownIndex].y,corners[leftDownIndex].x).val[0]<< "\n";

					frameCount++;
				}
			}

			else
			{
				cornerDepth.resize(4);

				//Calculate the average of depth values
				for(int i=0;i<leftUpDepth.size();i++)
				{
					cornerDepth[0]+=leftUpDepth[i];
					cornerDepth[1]+=rightUpDepth[i];
					cornerDepth[2]+=rightDownDepth[i];
					cornerDepth[3]+=leftDownDepth[i];
				}
				
				cornerDepth[0]/=leftUpDepth.size();
				cornerDepth[1]/=rightUpDepth.size();
				cornerDepth[2]/=rightDownDepth.size();
				cornerDepth[3]/=leftDownDepth.size();
				
				screenCatch=false;
				
				cornerCoordinates.resize(0);

				//Calculate the coordonates of four corners in the reference of the camera
				for(int i=0;i<screenCorners.size();i++)
				{
					XnPoint3D proj, real; 
					proj.X = screenCorners[i].x; 
					proj.Y = screenCorners[i].y; 
					proj.Z = cornerDepth[i]; 

					//The result of this function seems not correct, so I use another method to calculate the coordonates
					/*depthGenerator.ConvertProjectiveToRealWorld(1, &proj, &real); */
					float F=0.0019047619;
					real.X=(proj.X-320)*proj.Z*F;
					real.Y=(240-proj.Y)*proj.Z*F;
					real.Z=proj.Z;
					cornerCoordinates.push_back(cv::Point3f( real.X*0.001f, real.Y*0.001f, real.Z*0.001f)); // from mm to meters 
				}
			}

		}

		/// Show what you got
		cvNamedWindow("depth",1);  
		cv::imshow("depth",result);

		cvNamedWindow("image",1);  
		cv::imshow("image",color);

		cvNamedWindow("colorSegmentation",1);
		cv::imshow("colorSegmentation",colorGray);

		cv::Mat mix;
		addWeighted( depthNew, 0.8, drawing, 0.2, 0.0,mix);

		cvNamedWindow("Mix",1);
		cv::imshow("Mix",mix);

		namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
		imshow( "Hull demo", drawing );

		namedWindow( "corner", CV_WINDOW_AUTOSIZE );
		imshow( "corner", copy );

		cvNamedWindow("fullScreen",CV_WINDOW_NORMAL);
		cvSetWindowProperty("fullScreen",CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		cv::imshow("fullScreen",fullScreen);
		
		//////////////////
        key=cvWaitKey(20);  
    }  
  
    //destroy  
    cvDestroyWindow("depth");  
    cvDestroyWindow("image");  
	cvDestroyWindow("colorSegmentation");
	cvDestroyWindow("Mix");
	cvDestroyWindow("Hull demo");
	cvDestroyWindow("corner");
	cvDestroyWindow("fullScreen");
    cvReleaseImage(&imgDepth16u);  
    cvReleaseImage(&imgRGB8u);  
    cvReleaseImage(&depthShow);  
    cvReleaseImage(&imageShow);  
    context.StopGeneratingAll();  
    context.Shutdown();  
    return 0;  
}
