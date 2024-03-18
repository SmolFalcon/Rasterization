#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <list>
#include <ppl.h>
#include <mutex>

using namespace cv;
using namespace std;
using namespace concurrency;
mutex lockthread;


//struct to store triangle information in queue
struct triangle
{
	int a;
	int b;
	int c;

	double aUV[3];
	double bUV[3];
	double cUV[3];
};

//Normalize input pVec
void VectorNormalize(double pVec[])
{
	double magnitude = 0;

	for (int i = 0; i < 3; i++)
	{
		magnitude += pow(pVec[i], 2);
	}

	magnitude = sqrt(magnitude);

	for (int i = 0; i < 3; i++)
	{
		pVec[i] = (pVec[i] / magnitude);
	}
}

double DoubleRemap(double pInput, double pOldMin, double pOldMax, double pNewMin, double pNewMax)
{
	return ((pInput - pOldMin) / (pOldMax - pOldMin)) * (pNewMax - pNewMin) + pNewMin;
}
//Get direction vector from two points, A to B
void VectorAtoB(double pA[], double pB[], double pRes[], bool pNormalize = true)
{
	for (int i = 0; i < 3; i++)
	{
		pRes[i] = pB[i] - pA[i];
	}

	if (pNormalize)
	{
		VectorNormalize(pRes);
	}
}

double VectorDistance(double pA[], double pB[])
{
	double pRes[3];
	for (int i = 0; i < 3; i++)
	{
		pRes[i] = pB[i] - pA[i];
	}

	double magnitude = 0;

	for (int i = 0; i < 3; i++)
	{
		magnitude += pow(pRes[i], 2);
	}

	magnitude = sqrt(magnitude);

	return magnitude;
}

void VectorCrossProduct(double pA[],double pB[],double pRes[], bool pNormalize = true)
{
	pRes[0] = pA[1] * pB[2] - pA[2] * pB[1];
	pRes[1] = -(pA[0] * pB[2] - pA[2] * pB[0]);
	pRes[2] = pA[0] * pB[1] - pA[1] * pB[0];
	if (pNormalize)
	{
		VectorNormalize(pRes);
	}
}

double VectorDotProduct(double pA[], double pB[])
{
	double A[3] = { pA[0],pA[1] ,pA[2] };

	double B[3] = { pB[0],pB[1] ,pB[2] };

	VectorNormalize(A);
	VectorNormalize(B);

	return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
}

void Mat4x4MultPoint(double pMTC[4][4], double pPnt[], double pRes[])
{
	double lPnt[4] = { pPnt[0], pPnt[1] ,pPnt[2],1 }; //add extra dimension
	double lRes[4];

	for (int i = 0; i < 4; i++)
	{
		lRes[i] = pMTC[i][0] * lPnt[0] + pMTC[i][1] * lPnt[1] + pMTC[i][2] * lPnt[2] + pMTC[i][3] * lPnt[3];
	}
	
	pRes[0] = lRes[0];
	pRes[1] = lRes[1];
	pRes[2] = lRes[2];

	if (lRes[3] != 1)
	{
		pRes[0] = lRes[0] / lRes[3];
		pRes[1] = lRes[1] / lRes[3];
		pRes[2] = lRes[2] / lRes[3];
	}
}

void Mat3x3MultPoint(double pMTC[3][3], double pPnt[], double pRes[])
{
	double lPnt[3] = { pPnt[0], pPnt[1] ,pPnt[2]};
	double lRes[3];

	for (int i = 0; i < 3; i++)
	{
		lRes[i] = pMTC[i][0] * lPnt[0] + pMTC[i][1] * lPnt[1] + pMTC[i][2] * lPnt[2];
	}

	pRes[0] = lRes[0];
	pRes[1] = lRes[1];
	pRes[2] = lRes[2];
}

void DrawPoint(Mat pImg, double pPosXY[], int pValBGR[])
{
	int U, V;
	
	U = pPosXY[0];
	V = pPosXY[1];

	//0.5 -> floor 0.6 and up -> ceil
	U = (pPosXY[0] - floor(pPosXY[0]) > 0.5) ? ceil(pPosXY[0]) : floor(pPosXY[0]);
	V = (pPosXY[1] - floor(pPosXY[1]) > 0.5) ? ceil(pPosXY[1]) : floor(pPosXY[1]);



	//cout << pPosXY[0] << ", " << pPosXY[1] << endl;
	if (!(U >= pImg.cols || V >= pImg.rows || U < 0 || V < 0)) //Check if the point is within bounds
	{
		pImg.at<Vec3b>(V, U)[0] = pValBGR[0];
		pImg.at<Vec3b>(V, U)[1] = pValBGR[1];
		pImg.at<Vec3b>(V, U)[2] = pValBGR[2];
	}
}
//Print pVec in format x,y,z
void VectorPrint(double pVec[])
{
	cout << pVec[0] << "," << pVec[1] << "," << pVec[2] << endl;
}

double ProyectPoint3Dto2D(double pPoint3D[],double pPoint2D[], int pResolution[],double pMTC[4][4], double pTP[4][4])
{
	double pnt_vector[3] = { pPoint3D[0], pPoint3D[1], pPoint3D[2] };

	Mat4x4MultPoint(pMTC, pnt_vector, pnt_vector); //get point in camera coordinates
	double depth = pnt_vector[2];
	Mat4x4MultPoint(pTP, pnt_vector, pnt_vector); //get point in plane proyection coordinates

	//remap points to display space
	pnt_vector[0] = DoubleRemap(pnt_vector[0], 0.5, -0.5, 0, 1);
	pnt_vector[1] = DoubleRemap(pnt_vector[1], -0.5, 0.5, 0, 1);

	pPoint2D[0] = pnt_vector[0] * pResolution[0];
	pPoint2D[1] = pnt_vector[1] * pResolution[1];

	return depth;
	//if (!(Point2D[0] >= pResolution[0] || Point2D[1] >= pResolution[1] || Point2D[0] < 0 || Point2D[1] < 0)) //Check if all points are within bounds
	//{
	//	DrawPoint(object_mask, Point2D, val);
	//}
}
//Test if a point P is inside a triangle using barycentric coordinates
bool BarycentricTest(double pA[], double pB[], double pC[], double pP[],double BaryCoords[])
{
	double v0[2] = { pB[0] - pA[0],pB[1] - pA[1] };
	double v1[2] = { pC[0] - pA[0],pC[1] - pA[1] };
	double v2[2] = { pP[0] - pA[0],pP[1] - pA[1] };

	double d00 = v0[0] * v0[0] + v0[1] * v0[1];
	double d01 = v0[0] * v1[0] + v0[1] * v1[1];
	double d11 = v1[0] * v1[0] + v1[1] * v1[1];
	double d20 = v2[0] * v0[0] + v2[1] * v0[1];
	double d21 = v2[0] * v1[0] + v2[1] * v1[1];

	double denom = d00 * d11 - d01 * d01;
	double u, v, w;
	v = (d11 * d20 - d01 * d21) / denom;

	w = (d00 * d21 - d01 * d20) / denom;

	u = 1 - v - w;

	BaryCoords[0] = u;

	BaryCoords[1] = v;

	BaryCoords[2] = w;

	if (u >= 0 && v >= 0 && w >= 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void BarycentricRemap(double pA[], double pB[], double pC[], double pBaryCoords[], double pRes[])
{
	pRes[0] = pA[0] * pBaryCoords[0] + pB[0] * pBaryCoords[1] + pC[0] * pBaryCoords[2];
	pRes[1] = pA[1] * pBaryCoords[0] + pB[1] * pBaryCoords[1] + pC[1] * pBaryCoords[2];
	pRes[2] = pA[2] * pBaryCoords[0] + pB[2] * pBaryCoords[1] + pC[2] * pBaryCoords[2];
}

void DrawLine(Mat pImg, double pA[], double pB[],int pValBGR[])
{
	double mx = pA[0] - pB[0];
	double my = pA[1] - pB[1];
	double S = (abs(mx) > abs(my)) ? abs(mx) : abs(my); //shorthand if statement to get the largest of the two

	double delta_x = mx / S;
	double delta_y = my / S;

	double pixel[2] = { pB[0],pB[1] };

	while ( (abs(pixel[0] - pA[0]) + abs(pixel[1] - pA[1])) > 1  )
	{
		pixel[0] = pixel[0] + delta_x;
		pixel[1] = pixel[1] + delta_y;
		//cout << pixel[0] << ", " << pixel[1] << endl;
		DrawPoint(pImg, pixel, pValBGR);
	}

	DrawPoint(pImg, pA, pValBGR);

	DrawPoint(pImg, pB, pValBGR);
}

void RasterTriangle(triangle pTriangle, double pPointArr[][3], double pPointNormal[][3], double pA[], double pB[], double pC[], double pDepthA, double pDepthB, double pDepthC, Mat pTexture ,Mat pRender,Mat pObjectMask, Mat pDepthBuffer,Mat pW_Normal, Mat pW_Pos, list<double>* pZ_minmax)
{
	//Mat lTempLines = Mat::zeros(pImg.rows, pImg.cols, CV_8UC3); //image to isolate the lines of the triangle from the final image

	//Compute Screen space Normal
	double AB[3];
	double AC[3];
	double Normal[3];
	double nA[3] = { pA[0],pA[1],pDepthA };
	double nB[3] = { pB[0],pB[1],pDepthB };
	double nC[3] = { pC[0],pC[1],pDepthC };
	VectorAtoB(nA, nB, AB);
	VectorAtoB(nA, nC, AC);
	VectorCrossProduct(AB, AC, Normal);

	double pWA[3];
	double pWB[3];
	double pWC[3];
	
	pWA[0] = pPointArr[pTriangle.a][0];
	pWA[1] = pPointArr[pTriangle.a][1];
	pWA[2] = pPointArr[pTriangle.a][2];

	pWB[0] = pPointArr[pTriangle.b][0];
	pWB[1] = pPointArr[pTriangle.b][1];
	pWB[2] = pPointArr[pTriangle.b][2];

	pWC[0] = pPointArr[pTriangle.c][0];
	pWC[1] = pPointArr[pTriangle.c][1];
	pWC[2] = pPointArr[pTriangle.c][2];

	//Compute World Normal
	double w_Normal[3];
	VectorAtoB(pWA, pWB, AB);
	VectorAtoB(pWA, pWC, AC);
	VectorCrossProduct(AB, AC, w_Normal);




	//cout << Normal[0]<<"," <<Normal[1]<<","<<Normal[2] << endl;

	//DrawLine(lTempLines, pA, pB, val);

	//DrawLine(lTempLines, pA, pC, val);

	//DrawLine(lTempLines, pB, pC, val);

	//DrawLine(pImg, pA, pB, val);

	//DrawLine(pImg, pA, pC, val);

	//DrawLine(pImg, pB, pC, val);



	//Get max and min rows and cols to process a smaller window
	int minrows = pObjectMask.rows;
	int mincols = pObjectMask.cols;
	int maxrows = 0;
	int maxcols = 0;

	//minmax rows
	if (pA[1] > maxrows)
	{
		maxrows = pA[1];
	}
	if (pB[1] > maxrows)
	{
		maxrows = pB[1];
	}
	if (pC[1] > maxrows)
	{
		maxrows = pC[1];
	}

	if (pA[1] < minrows)
	{
		minrows = pA[1];
	}
	if (pB[1] < minrows)
	{
		minrows = pB[1];
	}
	if (pC[1] < minrows)
	{
		minrows = pC[1];
	}
	//minmax columns
	if (pA[0] > maxcols)
	{
		maxcols = pA[0];
	}
	if (pB[0] > maxcols)
	{
		maxcols= pB[0];
	}
	if (pC[0] > maxcols)
	{
		maxcols = pC[0];
	}

	if (pA[0] < mincols)
	{
		mincols = pA[0];
	}
	if (pB[0] < mincols)
	{
		mincols = pB[0];
	}
	if (pC[0] < mincols)
	{
		mincols = pC[0];
	}
	maxrows++;
	minrows--;

	maxcols++;
	mincols--;

	if (maxrows > pObjectMask.rows)
	{
		maxrows = pObjectMask.rows;
	}
	if (minrows < 0)
	{
		minrows = 0;
	}

	if (maxcols > pObjectMask.cols)
	{
		maxcols = pObjectMask.cols;
	}
	if (mincols < 0)
	{
		mincols = 0;
	}


	//parallel_for(int(minrows), maxrows, [&](size_t row)
	for (int row = minrows; row < maxrows; row++)
	{
		for (int col = mincols; col < maxcols; col++)
		{
			double lP[2] = { col,row }; //test point
			double lBaryCoords[3]; //array to store barycentric coordinates

			if (BarycentricTest(pA, pB, pC, lP,lBaryCoords))//Test to check if the point is inside triangle
			//if(lTemp.at<uchar>(row, col)>0)
			{
				pObjectMask.at<float>(row, col) = 1;

				//Compute Depth	
				int sign = (Normal[2] > 0) ? 1 : -1;
				double div = (abs(Normal[2]) < 0.01) ? 0.01 * sign : Normal[2];

				double Z = ((Normal[0] * col + Normal[1] * row) - (Normal[0] * pA[0] + Normal[1] * pA[1] + Normal[2] * pDepthA)) / div;

				//lockthread.lock();//lock thread to avoid errors
				pZ_minmax->push_front(Z); //store depth in list
				//lockthread.unlock();

				//Check depth before placing pixels
				if (pDepthBuffer.at<float>(row, col) < Z || pDepthBuffer.at<float>(row, col) == 0)
				{
					pDepthBuffer.at<float>(row, col) = Z;
					
					//World space normal (BGR)

					double lTemp[3];

					BarycentricRemap(pPointNormal[pTriangle.a], pPointNormal[pTriangle.b], pPointNormal[pTriangle.c], lBaryCoords, lTemp);
					
					VectorNormalize(lTemp);

					pW_Normal.at<Vec3f>(row, col)[0] = lTemp[2];
					pW_Normal.at<Vec3f>(row, col)[1] = lTemp[1];
					pW_Normal.at<Vec3f>(row, col)[2] = lTemp[0];

					//pW_Normal.at<Vec3f>(row, col)[0] = w_Normal[2];
					//pW_Normal.at<Vec3f>(row, col)[1] = w_Normal[1];
					//pW_Normal.at<Vec3f>(row, col)[2] = w_Normal[0];

					//Render texture
					double TexColor[3];
					double A[3] = { pTriangle.aUV[0],pTriangle.aUV[1],0 };
					double B[3] = { pTriangle.bUV[0],pTriangle.bUV[1],0 };
					double C[3] = { pTriangle.cUV[0],pTriangle.cUV[1],0 };

					BarycentricRemap(A, B, C, lBaryCoords, TexColor);

					int lrow = TexColor[1] * pTexture.rows;
					int lcol = TexColor[0] * pTexture.cols;
					//cout << lrow << "." << lcol << endl;
					pRender.at<Vec3f>(row, col)[0] = pTexture.at<Vec3b>(lrow, lcol)[0];
					pRender.at<Vec3f>(row, col)[1] = pTexture.at<Vec3b>(lrow, lcol)[1];
					pRender.at<Vec3f>(row, col)[2] = pTexture.at<Vec3b>(lrow, lcol)[2];


					//Remap coordinates to get world position
					double WorldPos[3];
					BarycentricRemap(pWA, pWB, pWC, lBaryCoords,WorldPos);


					//World position (BGR)
					pW_Pos.at<Vec3f>(row, col)[0] = WorldPos[2];
					pW_Pos.at<Vec3f>(row, col)[1] = WorldPos[1];
					pW_Pos.at<Vec3f>(row, col)[2] = WorldPos[0];

					////Render
					//double LightPos[3] = { 2.5,3,-0.5 };
					//double SurfaceToLight[3];
					//double SurfaceToCam[3];

					//VectorAtoB(WorldPos, LightPos, SurfaceToLight);
					//VectorAtoB(WorldPos, pCamPos, SurfaceToCam);

					//double DotNormalLight = VectorDotProduct(w_Normal, SurfaceToLight);
					//double ldiff = (DotNormalLight > 0) ? DotNormalLight : 0;


					//double DotNormalCam = VectorDotProduct(w_Normal, SurfaceToCam);

					//double lspec = ((DotNormalLight * DotNormalCam ) > 0) ? (DotNormalLight*DotNormalCam) : 0;

					//pDiff.at<float>(row, col) = pow(lspec,3) + ldiff * 0.5 * 0;
				}
			}
		}
	}//);
}

void GetTriangleNormal(triangle pTriangle,double pPoints[][3], double pOutput[])
{
	double A[3];
	A[0] = pPoints[pTriangle.a][0];
	A[1] = pPoints[pTriangle.a][1];
	A[2] = pPoints[pTriangle.a][2];

	double B[3];
	B[0] = pPoints[pTriangle.b][0];
	B[1] = pPoints[pTriangle.b][1];
	B[2] = pPoints[pTriangle.b][2];

	double C[3];
	C[0] = pPoints[pTriangle.c][0];
	C[1] = pPoints[pTriangle.c][1];
	C[2] = pPoints[pTriangle.c][2];

	double AB[3];
	VectorAtoB(A, B, AB);
	double AC[3];
	VectorAtoB(A, C, AC);
	
	double Cross[3];

	VectorCrossProduct(AB, AC, Cross);

	pOutput[0] = Cross[0];
	pOutput[1] = Cross[1];
	pOutput[2] = Cross[2];
}

int main()
{
	string lfile_line;//token string
	ifstream lfile("config.txt");//open text file

	double CamPos[3];
	double CamTarget[3];
	double CamUpVec[3];
	double CamFOV[2];
	int CamRes[2];
	int idx = 0;
	int pointCount = 0;
	bool StopFlag = false;

	Mat Texture = cv::imread("tex.png");



	//Store values for CamPos, CamTarget and CamUpVec from the config file
	while (getline(lfile, lfile_line)&&!StopFlag) 
	{
		if (lfile_line.find(':') != string::npos) //search string in line
		{
			string DataValue;
			stringstream str(lfile_line);
			while (getline(str, DataValue, ':')) //separate name from value
			{
				if (DataValue.find("Cam") == string::npos) //discard value name
				{
					string point;
					stringstream temp_line(DataValue);
					int pidx = 0;

					if (idx == 0)
					{
						while (getline(temp_line, point, ',')) //iterate each component of point vector
						{
							CamPos[pidx] = stod(point);//store vector component value in array
							pidx++;
						}
						
					}
					else if (idx == 1)
					{
						while (getline(temp_line, point, ',')) //iterate each component of point vector
						{
							CamTarget[pidx] = stod(point);//store vector component value in array
							pidx++;
						}
					}
					else if (idx == 2)
					{
						while (getline(temp_line, point, ',')) //iterate each component of point vector
						{
							CamUpVec[pidx] = stod(point);//store vector component value in array	
							pidx++;
						}
					}
					else if (idx == 3)
					{
						while (getline(temp_line, point, ',')) //iterate each component of point vector
						{
							CamFOV[pidx] = stod(point);//store vector component value in array	
							pidx++;
						}
					}
					else if (idx == 4)
					{
						while (getline(temp_line, point, ',')) //iterate each component of point vector
						{
							CamRes[pidx] = stod(point);//store vector component value in array	
							pidx++;
						}
					}
					idx++;
				}
			}
		}
		//Store point count
		if (lfile_line.find("Points") != string::npos)
		{
			string DataValue;
			stringstream str(lfile_line);
			while (getline(str, DataValue, ':')) //separate name from value
			{
				if (DataValue.find("Points") == string::npos) //discard value name
				{
					pointCount = stoi(DataValue);
					StopFlag = true;
				}
			}
		}
	}


	//store points in array
	ifstream points_file("config.txt");//open text file
	auto pointArr = new double[pointCount][3];
	int rowidx = 0;






	ifstream triangles_file("config.txt");//open text file
	bool flag = false; //flag to start reading triangle section of the file
	bool flagUV = false; //flag to start reading UV section of the file
	bool flagLight = false; //flag to start reading Light section of the file
	queue<triangle> triangles; //queue to store triangles

	queue<triangle>* pointTris = new queue<triangle>[pointCount]; //Array of queues to store all the triangles sharing a point


	//////////Lights/////////////////
	struct Light
	{
		double LightPos[3];
	};

	queue<Light> Lights;

	//Lights.push(Light_1);
	//Lights.push(Light_2);




	int progresscount = 0;
	int progress_factor = pointCount / 10;
	progress_factor = (progress_factor == 0) ? 1 : progress_factor;
	while (getline(points_file, lfile_line)) //iterate each line of the file
	{
		double pnt_vector[3]; //array to store vector
		string point;
		stringstream temp_line(lfile_line);
		int colidx = 0;
		if (lfile_line.find(':') == string::npos && rowidx<pointCount) //search : in line to skip data lines
		{
			while (getline(temp_line, point, ',')) //iterate each component of point vector
			{
				pointArr[rowidx][colidx] = stod(point);//store vector component value in array
				colidx++;
			}
			rowidx++;
			progresscount++;
		}		


		//Store triangle information
		string tri;
		int idx = 0;

		if (lfile_line.find("Triangles:") != string::npos)
		{
			flag = true;
		}

		if (lfile_line.find(':') == string::npos && flag && !flagLight) //search : in line to skip data lines
		{
			triangle temp_tri;
			while (getline(temp_line, tri, ',')) //iterate each component of point vector
			{
				//cout << tri << endl;
				if (idx == 0)
				{
					temp_tri.a = stoi(tri);
				}
				if (idx == 1)
				{
					temp_tri.b = stoi(tri);
				}
				if (idx == 2)
				{
					temp_tri.c = stoi(tri);
				}
				if (idx == 3)
				{
					temp_tri.aUV[0] = stod(tri);
				}
				if (idx == 4)
				{
					temp_tri.aUV[1] = stod(tri);
				}
				if (idx == 5)
				{
					temp_tri.aUV[2] = stod(tri);
				}
				if (idx == 6)
				{
					temp_tri.bUV[0] = stod(tri);
				}
				if (idx == 7)
				{
					temp_tri.bUV[1] = stod(tri);
				}
				if (idx == 8)
				{
					temp_tri.bUV[2] = stod(tri);
				}
				if (idx == 9)
				{
					temp_tri.cUV[0] = stod(tri);
				}
				if (idx == 10)
				{
					temp_tri.cUV[1] = stod(tri);
				}
				if (idx == 11)
				{
					temp_tri.cUV[2] = stod(tri);
				}
				idx++;
			}
			triangles.push(temp_tri);
			//cout << temp_tri.a << "," << temp_tri.b << "," << temp_tri.c << endl;
			pointTris[temp_tri.a].push(temp_tri);
			pointTris[temp_tri.b].push(temp_tri);
			pointTris[temp_tri.c].push(temp_tri);
		}

		if (lfile_line.find("Lights:") != string::npos)
		{
			flagLight = true;
		}

		if (lfile_line.find(':') == string::npos && flagLight) //search : in line to skip data lines
		{
			Light temp_light;
			int idx = 0;
			while (getline(temp_line, tri, ',')) //iterate each component of point vector
			{
				
				//cout << tri << endl;
				if (idx == 0)
				{
					temp_light.LightPos[0] = stod(tri);
				}
				if (idx == 1)
				{
					temp_light.LightPos[1] = stod(tri);
				}
				if (idx == 2)
				{
					temp_light.LightPos[2] = stod(tri);
				}
				idx++;
			}

			cout << temp_light.LightPos[0] << ',' << temp_light.LightPos[1] << ',' << temp_light.LightPos[2] << endl;
			Lights.push(temp_light);
			
		}



		if (progresscount % progress_factor == 0 && progresscount < pointCount)
		{
			cout << progresscount<< endl;
		}	
	}



	auto pointNormal = new double[pointCount][3];
	/////Compute point normals//////
	for (int i = 0; i < pointCount; i++)
	{
		double NormalSum[3] = { 0,0,0 };
		while (!pointTris[i].empty())
		{
			double lNormal[3];
			GetTriangleNormal(pointTris[i].front(), pointArr, lNormal);
		
			NormalSum[0] += lNormal[0];
			NormalSum[1] += lNormal[1];
			NormalSum[2] += lNormal[2];

			pointTris[i].pop();
		}

		pointNormal[i][0] = NormalSum[0];
		pointNormal[i][1] = NormalSum[1];
		pointNormal[i][2] = NormalSum[2]; 

		VectorNormalize(pointNormal[i]);
		//VectorPrint(pointNormal[i]);
	}





	//////////Camera///////////


	double CamU[3];
	double CamV[3];
	double CamW[3];
	double temp[3];

	//CamV[0] = CamUpVec[0];
	//CamV[1] = CamUpVec[1];
	//CamV[2] = CamUpVec[2];

	//vector from camera to target and from camera to Up vector
	VectorAtoB(CamPos, CamTarget, CamW);
	VectorAtoB(CamPos, CamUpVec, temp); 

	VectorCrossProduct(temp, CamW, CamU);
	VectorCrossProduct(CamW, CamU, CamV);

	//VectorPrint(CamU);
	//VectorPrint(CamV);
	//VectorPrint(CamW);

	double Mct[3][3] = {{CamU[0],CamU[1],CamU[2]},
						{CamV[0],CamV[1],CamV[2]},
						{CamW[0],CamW[1],CamW[2]}
						}; //Transposed camera matrix

	double CamTransform[3];

	double NegCamPos[3] = { -CamPos[0],-CamPos[1],-CamPos[2] };

	Mat3x3MultPoint(Mct, NegCamPos, CamTransform);

	//Camera transform matrix with offset
	double Mct_Offset[4][4] = { {CamU[0],CamU[1],CamU[2],CamTransform[0]},
								{CamV[0],CamV[1],CamV[2],CamTransform[1]},
								{CamW[0],CamW[1],CamW[2],CamTransform[2]},
								{0,0,0,1} };

	double l = 1000;
	double c = 0;

	double a = -((l + c) / (l - c));
	double b = -((2 * (l * c)) / (l - c));
	//a = 1;
	//b = 0;

	double f = 1;

	//calculate plane width and height from field of vision angles
	double w = (tan(CamFOV[0] / 2) * f) * 2;

	double h = (tan(CamFOV[1] / 2) * f) * 2;

	//Build proyection plane matrix
	double Tp[4][4] = { {f/w,0,0,0},
						{0,f/h,0,0},
						{0,0,a,b},
						{0,0,-1/f,0}
	};








	//////////Raster and Render////////////
	double var = 0;
	int framecount = 0;
	while (framecount < 1)
	{	
		framecount++;
		/*framecount++;
		var=1;
		double TransformY[3][3] = { {cos(var),0,sin(var)},
									{0,1,0},
									{-sin(var),0,cos(var)}};

		double TransformX[3][3] = { {1,0,0},
									{0,cos(var),-sin(var)},
									{0,sin(var),cos(var)} };

*/


		Mat render = Mat::zeros(CamRes[1], CamRes[0], CV_32FC3);
		Mat object_mask = Mat::zeros(CamRes[1], CamRes[0], CV_32F);
		Mat z_buffer = Mat::zeros(CamRes[1], CamRes[0], CV_32F);
		Mat diffuse = Mat::zeros(CamRes[1], CamRes[0], CV_32F);
		Mat specular = Mat::zeros(CamRes[1], CamRes[0], CV_32F);
		Mat ambient = Mat::zeros(CamRes[1], CamRes[0], CV_32F);
		Mat world_normal = Mat::zeros(CamRes[1], CamRes[0], CV_32FC3);
		Mat world_position = Mat::zeros(CamRes[1], CamRes[0], CV_32FC3);

		list<double> z_minmax;

		int cout_count = 0;
		int cout_factor = triangles.size() / 10;
		cout_factor = (cout_factor == 0) ? 1 : cout_factor;

		//copy queues
		queue<triangle> ltriangles(triangles);
		queue<Light> lLights(Lights);

		//Transform Point and Normal
		//for (int i = 0; i < pointCount; i++)
		//{
		//	double temp[3] = { pointArr[i][0], pointArr[i][1], pointArr[i][2] };
		//	double temp_n[3] = { pointNormal[i][0], pointNormal[i][1], pointNormal[i][2] };

		//	Mat3x3MultPoint(TransformX, temp, pointArr[i]);
		//	Mat3x3MultPoint(TransformX, temp_n, pointNormal[i]);
		//}

		while (!ltriangles.empty())
		{
			triangle tri_abc = ltriangles.front(); //get the current triangle
			int val[3] = { 255,255,255 }; //color of the pixel
			//3D positions of the points
			double pnt_a[3] = { pointArr[tri_abc.a][0], pointArr[tri_abc.a][1], pointArr[tri_abc.a][2] };
			double pnt_b[3] = { pointArr[tri_abc.b][0], pointArr[tri_abc.b][1], pointArr[tri_abc.b][2] };
			double pnt_c[3] = { pointArr[tri_abc.c][0], pointArr[tri_abc.c][1], pointArr[tri_abc.c][2] };

			//2D positions of the points and depth from camera
			double depth_a, depth_b, depth_c;
			double a[2];
			depth_a = ProyectPoint3Dto2D(pnt_a, a, CamRes, Mct_Offset, Tp);
			double b[2];
			depth_b = ProyectPoint3Dto2D(pnt_b, b, CamRes, Mct_Offset, Tp);
			double c[2];
			depth_c = ProyectPoint3Dto2D(pnt_c, c, CamRes, Mct_Offset, Tp);


			RasterTriangle(tri_abc, pointArr, pointNormal, a, b, c, depth_a, depth_b, depth_c, Texture, render, object_mask, z_buffer, world_normal, world_position, &z_minmax);

			//DrawPoint(object_mask, a, val);
			//
			//DrawPoint(object_mask, b, val);

			//DrawPoint(object_mask, c, val);

			ltriangles.pop();
			if (cout_count % cout_factor == 0)
			{
				cout << "Triangles remaining: " << triangles.size() << endl;
			}
			cout_count++;
		}

		//Remap the range of the depth map
		z_minmax.sort();

		double z_min = z_minmax.front();
		double z_max = z_minmax.back();

		for (int row = 0; row < CamRes[1]; row++)
		{
			for (int col = 0; col < CamRes[0]; col++)
			{
				if (z_buffer.at<float>(row, col) == 0)
				{
					z_buffer.at<float>(row, col) = z_min * 0;
				}
				else
				{
					z_buffer.at<float>(row, col) = DoubleRemap(z_buffer.at<float>(row, col), z_min, z_max, 0, 1);
				}
			}
		}








		//////////////Render//////////////



		while (!lLights.empty())
		{
			for (int row = 0; row < CamRes[1]; row++)
			{
				for (int col = 0; col < CamRes[0]; col++)
				{
					if (object_mask.at<float>(row, col) > 0)
					{
						double SurfaceToLight[3];
						double SurfaceToCam[3];
						double WorldPos[3];
						double w_Normal[3];

						WorldPos[0] = world_position.at<Vec3f>(row, col)[2];
						WorldPos[1] = world_position.at<Vec3f>(row, col)[1];
						WorldPos[2] = world_position.at<Vec3f>(row, col)[0];

						w_Normal[0] = world_normal.at<Vec3f>(row, col)[2];
						w_Normal[1] = world_normal.at<Vec3f>(row, col)[1];
						w_Normal[2] = world_normal.at<Vec3f>(row, col)[0];

						VectorAtoB(WorldPos, lLights.front().LightPos, SurfaceToLight);
						VectorAtoB(WorldPos, CamPos, SurfaceToCam);

						VectorNormalize(SurfaceToLight);
						VectorNormalize(SurfaceToCam);

						double LightDistance = VectorDistance(WorldPos, Lights.front().LightPos);

						double DotNormalLight = VectorDotProduct(w_Normal, SurfaceToLight);

						double DotNormalCam = VectorDotProduct(w_Normal, SurfaceToCam);

						//Diffuse
						double ldiff = (DotNormalLight > 0) ? DotNormalLight : 0;

						diffuse.at<float>(row, col) = diffuse.at<float>(row, col) + DotNormalLight / pow(LightDistance, 2) * 5;

						//Specular
						double lspec = ((DotNormalLight * DotNormalCam) > 0) ? (DotNormalLight * DotNormalCam) : 0;

						specular.at<float>(row, col) = specular.at<float>(row, col) + (pow(lspec, 2) / pow(LightDistance, 2)) * 10;

						//Ambient
						if (object_mask.at<float>(row, col) > 0)
						{
							ambient.at<float>(row, col) = 0.2;
						}
					}
				}
			}
			lLights.pop();
		}





		diffuse = diffuse + ambient;

		Mat B, G, R;
		vector<Mat> channels(3);
		cv::split(render, channels);

		//diffuse = (channels[0]/255);

		cv::multiply(channels[0] / 255, diffuse, channels[0]);
		cv::multiply(channels[1] / 255, diffuse, channels[1]);
		cv::multiply(channels[2] / 255, diffuse, channels[2]);

		channels[0] = (channels[0] + specular) * 255;
		channels[1] = (channels[1] + specular) * 255;
		channels[2] = (channels[2] + specular) * 255;

		cv::merge(channels, render);








		//cv::imshow("Object mask", object_mask);
		//cv::imshow("Z Buffer", z_buffer);
		//cv::normalize(world_normal, world_normal, 0, 1, NORM_MINMAX);
		//cv::imshow("World space normal", world_normal);
		//cv::imshow("Texture", Texture);

		//cv::normalize(world_position, world_position, 0, 1, NORM_MINMAX);
		//cv::imshow("World position", world_position);

		//cv::normalize(diffuse, diffuse, 0, 1, NORM_MINMAX);
		//cv::imshow("Diffuse", diffuse);

		//cv::normalize(render, render, 0, 1, NORM_MINMAX);
		cv::imshow("Render", render/255);


		//cv::imwrite("ObjectMask.tiff", object_mask);
		//cv::imwrite("Zbuffer.tiff", z_buffer);
		//cv::imwrite("WorldSpaceNormal.tiff", world_normal);
		//cv::imwrite("World position.tiff", world_position);
		//cv::normalize(render, render, 0, 255, NORM_RELATIVE);
		cv::imwrite("Render_"+ to_string(framecount) +".png", render);

		
		cv::waitKey(0);
	}

	delete[] pointArr;
	delete[] pointNormal;
	delete[] pointTris;
	return 0;
}