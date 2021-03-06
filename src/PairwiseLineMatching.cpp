/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2011-2012, Lilian Zhang, all rights reserved.
Copyright (C) 2016, Malte Splietker, all rights reserved.
Third party copyrights are property of their respective owners.

To extract edge and lines, this library implements the EDLines Algorithm and the Edge Drawing detector:
http://www.sciencedirect.com/science/article/pii/S0167865511001772
http://www.sciencedirect.com/science/article/pii/S1047320312000831

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "lbd_descriptor/PairwiseLineMatching.hh"

#include <arlsmat.h>
#include <arlssym.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>


namespace lbd_descriptor
{

using namespace std;
using cv::Mat;
using cv::norm;

const double PI = M_PI;
const double TWO_PI = 2 * PI;

const double IntersectionRationDifThreshold = 1;
const double ProjectionRationDifThreshold = 1;


// This is used when getting matching results from principal eigen vector
const double WeightOfMeanEigenVec = 0.1;

void PairwiseLineMatching::LineMatching(ScaleLines &linesInLeft, ScaleLines &linesInRight,
                                        std::vector<unsigned int> &matchResult)
{
  //compute the global rotation angle of image pair;
  globalRotationAngle_ = GlobalRotationOfImagePair_(linesInLeft, linesInRight);
  BuildAdjacencyMatrix_(linesInLeft, linesInRight);
  MatchingResultFromPrincipalEigenvector_(linesInLeft, linesInRight, matchResult);
}


double PairwiseLineMatching::GlobalRotationOfImagePair_(ScaleLines &linesInLeft, ScaleLines &linesInRight)
{
  double rotationAngle = 0;

  // Step 1: compute the angle histogram of lines in the left and right images
  unsigned int dim = (unsigned int) (360 /
                                     parameters_.globalRotationHistogramResolutionScale); //number of the bins of histogram
  unsigned int index; //index in the histogram
  double direction;
  double scalar = 180 / (parameters_.globalRotationHistogramResolutionScale * PI);//used when compute the index
  double angleShift =
      (parameters_.globalRotationHistogramResolutionScale * PI) / 360;//make sure zero is the middle of the interval

  Mat angleHistLeft = Mat::zeros(1, dim, CV_64F);
  Mat angleHistRight = Mat::zeros(1, dim, CV_64F);
  Mat lengthLeft = Mat::zeros(1, dim,
                              CV_64F); //lengthLeft[i] store the total line length of all the lines in the ith angle bin.
  Mat lengthRight = Mat::zeros(1, dim, CV_64F);

  for (unsigned int linenum = 0; linenum < linesInLeft.size(); linenum++)
  {
    direction = linesInLeft[linenum][0].direction + PI + angleShift;
    direction = direction < TWO_PI ? direction : (direction - TWO_PI);
    index = (unsigned int) floor(direction * scalar);
    angleHistLeft.at<double>(0, index) += 1;
    lengthLeft.at<double>(0, index) += linesInLeft[linenum][0].lineLength;
  }
  for (unsigned int linenum = 0; linenum < linesInRight.size(); linenum++)
  {
    direction = linesInRight[linenum][0].direction + PI + angleShift;
    direction = direction < TWO_PI ? direction : (direction - TWO_PI);
    index = (unsigned int) floor(direction * scalar);
    angleHistRight.at<double>(0, index) += 1;
    lengthRight.at<double>(0, index) += linesInRight[linenum][0].lineLength;
  }

  angleHistLeft /= norm(angleHistLeft);
  angleHistRight /= norm(angleHistRight);
  lengthLeft /= norm(lengthLeft);
  lengthRight /= norm(lengthRight);

  // Step 2: find shift to decide the approximate global rotation
  vector<double> difVec(dim); // The difference vector between left histogram and shifted right histogram
  double minDif = 10; // The minimal angle histogram difference
  double secondMinDif = 10; // The second minimal histogram difference
  unsigned int minShift = 0; // The shift of right angle histogram when minimal difference achieved

  vector<double> lengthDifVec(dim); // The length difference vector between left and right
  double minLenDif = 10; // The minimal length difference
  double secondMinLenDif = 10; // The second minimal length difference

  double normOfVec;
  for (unsigned int shift = 0; shift < dim; shift++)
  {
    for (unsigned int j = 0; j < dim; j++)
    {
      index = j + shift;
      index = index < dim ? index : (index - dim);
      difVec[j] = angleHistLeft.at<double>(j) - angleHistRight.at<double>(index);
      lengthDifVec[j] = lengthLeft.at<double>(j) - lengthRight.at<double>(index);
    }
    // Find the minShift and secondMinShift for angle histogram
    normOfVec = norm(difVec);
    if (normOfVec < secondMinDif)
    {
      if (normOfVec < minDif)
      {
        secondMinDif = minDif;
        minDif = normOfVec;
        minShift = shift;
      }
      else
      {
        secondMinDif = normOfVec;
      }
    }
    // Find the minLenShift and secondMinLenShift of length vector
    normOfVec = norm(lengthDifVec);
    if (normOfVec < secondMinLenDif)
    {
      if (normOfVec < minLenDif)
      {
        secondMinLenDif = minLenDif;
        minLenDif = normOfVec;
      }
      else
      {
        secondMinLenDif = normOfVec;
      }
    }
  }

  // First check whether there exist an approximate global rotation angle between image pair
  if (minDif < parameters_.globalRotationAcceptableHistogramAngle and
      minLenDif < parameters_.globalRotationAcceptableLengthDifference)
  {
    rotationAngle = minShift * parameters_.globalRotationHistogramResolutionScale;
    if (rotationAngle > 90 && 360 - rotationAngle > 90)
    {
      // In most case we believe the rotation angle between two image pairs should belong to [-Pi/2, Pi/2]
      rotationAngle = rotationAngle - 180;
    }
    rotationAngle = rotationAngle * PI / 180;
  }

#ifdef DEBUG_OUTPUT
  cout << "minimal histgram distance = " << minDif << ", Approximate global rotation angle = " << rotationAngle << endl;
#endif // #ifdef DEBUG_OUTPUT
  return rotationAngle;
}


void PairwiseLineMatching::BuildAdjacencyMatrix_(ScaleLines &linesInLeft, ScaleLines &linesInRight)
{
  unsigned int numLineLeft = (unsigned int) linesInLeft.size();
  unsigned int numLineRight = (unsigned int) linesInRight.size();

  /* Step 1: find nodes which are possible correspondent lines in the left and right images according to
   * their direction, gray value and gradient magnitude. */
  nodesList_.clear();

  if (linesInLeft.empty() or linesInRight.empty())
  {
    return;
  }

  unsigned long dimOfDes = linesInLeft[0][0].descriptor.size();
  Mat desDisMat(numLineLeft, numLineRight, CV_64F);

// //store descriptor for debug
//	Matrix<double> desCripLeft(numLineLeft,dimOfDes);
//	Matrix<double> desCripRight(numLineRight,dimOfDes);
//	for(unsigned int i=0; i<numLineLeft; i++){
//		for(unsigned int j=0; j<dimOfDes; j++){
//			desCripLeft[i][j] = linesInLeft[i].decriptor[j];
//		}
//	}
//	for(unsigned int i=0; i<numLineRight; i++){
//		for(unsigned int j=0; j<dimOfDes; j++){
//			desCripRight[i][j] = linesInRight[i].decriptor[j];
//		}
//	}
//	desCripLeft.Save("DescriptorLeft.txt");
//	desCripRight.Save("DescriptorRight.txt");

  // Compute descriptor distances.
  float *desL, *desR, *desMax, *desOld;
  float minDis, dis, temp;
  for (unsigned int idL = 0; idL < numLineLeft; idL++)
  {
    unsigned long sameLineSize = linesInLeft[idL].size();
    for (unsigned int idR = 0; idR < numLineRight; idR++)
    {
      minDis = 100;
      unsigned long sameLineSizeR = linesInRight[idR].size();
      for (short lineIDInSameLines = 0; lineIDInSameLines < sameLineSize; lineIDInSameLines++)
      {
        desOld = linesInLeft[idL][lineIDInSameLines].descriptor.data();
        for (unsigned long lineIDInSameLinesR = 0; lineIDInSameLinesR < sameLineSizeR; lineIDInSameLinesR++)
        {
          desL = desOld;
          desR = linesInRight[idR][lineIDInSameLinesR].descriptor.data();
          desMax = desR + dimOfDes;
          dis = 0;
          while (desR < desMax)
          {
            temp = *(desL++) - *(desR++);
            dis += temp * temp;
          }
          dis = sqrt(dis);
          if (dis < minDis)
          {
            minDis = dis;
          }
        }
      }
      desDisMat.at<double>(idL, idR) = minDis;
    }
  }
  // Filter out length- and angular outliers.
  for (unsigned int i = 0; i < numLineLeft; i++)
  {
    for (unsigned int j = 0; j < numLineRight; j++)
    {
      if (desDisMat.at<double>(i, j) > parameters_.descriptorDifferenceThreshold)
      {
        continue; // The descriptor difference is too large.
      }

      double lengthDiff = abs(linesInLeft[i][0].lineLength - linesInRight[j][0].lineLength) /
                          MIN(linesInLeft[i][0].lineLength, linesInRight[j][0].lineLength);
      if (lengthDiff > parameters_.lengthDifferenceThreshold)
      {
        continue; // The length difference is too large.
      }

      double angleDiff = abs(linesInLeft[i][0].direction + globalRotationAngle_ - linesInRight[j][0].direction);
      if (abs(TWO_PI - angleDiff) > parameters_.angleDifferenceThreshold &&
          angleDiff > parameters_.angleDifferenceThreshold)
      {
        continue; // The angle difference is too large.
      }

      // Line i in left image and line j in right image pass the test, (i,j) is a possible matched line pair.
      LBDNode node;
      node.leftLineID = i;
      node.rightLineID = j;
      nodesList_.push_back(node);
    }
  }
#ifdef DEBUG_OUTPUT
  cout << "the number of possible matched line pair = " << nodesList_.size() << endl;
//	desDisMat.Save("DescriptorDis.txt");
#endif // #ifdef DEBUG_OUTPUT

  // Step 2: step, build the adjacency matrix which reflect the geometric constraints between nodes.
  // The matrix is stored in the Compressed Sparse Column(CSC) format.
  unsigned int dim = (unsigned int) nodesList_.size(); // Dimension of the problem.
  int nnz = 0; // Number of non-zero elements in adjacenceMat.
  // adjacenceVec only store the lower part of the adjacency matrix which is a symmetric matrix.
  //                    | 0  1  0  2  0 |
  //                    | 1  0  3  0  1 |
  // eg:  adjMatrix =   | 0  3  0  2  0 |
  //                    | 2  0  2  0  3 |
  //                    | 0  1  0  3  0 |
  // adjacenceVec = [0,1,0,2,0,0,3,0,1,0,2,0,0,3,0]

  vector<double> adjacenceVec(dim * (dim + 1) / 2, 0.0);
  // In order to save computational time, the following variables are used to store
  // the pairwise geometric information which has been computed and will be reused many times
  // latter. The reduction of computational time is at the expenses of memory consumption.

  // Matrix of flags to show whether the ith pair of the left/right image has already been computed.
  vector<bool> computedLeft(numLineLeft * numLineLeft, false);
  vector<bool> computedRight(numLineRight * numLineRight, false);

  // The ratio of intersection point and the line in the left/right pair
  Mat intersecRatioLeft(numLineLeft, numLineLeft, CV_64F);
  Mat intersecRatioRight(numLineRight, numLineRight, CV_64F);

  // The point to line distance divided by the projected length of line in the left/right pair.
  Mat projRatioLeft(numLineLeft, numLineLeft, CV_64F);
  Mat projRatioRight(numLineRight, numLineRight, CV_64F);

  double iRatio1L, iRatio1R, iRatio2L, iRatio2R;
  double pRatio1L, pRatio1R, pRatio2L, pRatio2R;

  double iRatioDif, pRatioDif;

  double a1b2_a2b1; // a1b2 - a2b1
  double length1, length2, len;
  double similarity;
  for (unsigned int j = 0; j < dim; j++)
  { // Column
    unsigned int idLeft1 = nodesList_[j].leftLineID;
    unsigned int idRight1 = nodesList_[j].rightLineID;
    for (unsigned int i = j + 1; i < dim; i++)
    { // Row
      unsigned int idLeft2 = nodesList_[i].leftLineID;
      unsigned int idRight2 = nodesList_[i].rightLineID;
      if ((idLeft1 == idLeft2) || (idRight1 == idRight2))
      {
        continue; // Not satisfy the one to one match condition
      }
      // First compute the relative angle between left pair and right pair.
      double relativeAngleLeft = linesInLeft[idLeft1][0].direction - linesInLeft[idLeft2][0].direction;
      relativeAngleLeft = (relativeAngleLeft < PI) ? relativeAngleLeft : (relativeAngleLeft - TWO_PI);
      relativeAngleLeft = (relativeAngleLeft > (-PI)) ? relativeAngleLeft : (relativeAngleLeft + TWO_PI);
      double relativeAngleRight = linesInRight[idRight1][0].direction - linesInRight[idRight2][0].direction;
      relativeAngleRight = (relativeAngleRight < PI) ? relativeAngleRight : (relativeAngleRight - TWO_PI);
      relativeAngleRight = (relativeAngleRight > (-PI)) ? relativeAngleRight : (relativeAngleRight + TWO_PI);
      double relativeAngleDif = abs(relativeAngleLeft - relativeAngleRight);
      if ((TWO_PI - relativeAngleDif) > parameters_.angleDifferenceThreshold and
          relativeAngleDif > parameters_.angleDifferenceThreshold)
      {
        continue; // The relative angle difference is too large;
      }
      else if ((TWO_PI - relativeAngleDif) < parameters_.angleDifferenceThreshold)
      {
        relativeAngleDif = TWO_PI - relativeAngleDif;
      }

      /* At last, check the intersect point ratio and point to line distance ratio.
       * Check whether the geometric information of pairs (idLeft1,idLeft2) and (idRight1,idRight2)
       * have already been computed.
       */
      if (not computedLeft.at(idLeft1 * numLineLeft + idLeft2))
      { // Have not been computed yet
        /* compute the intersection point of segment i and j.
         * a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0.
         * x = (c2b1 - c1b2)/(a1b2 - a2b1) and
         * y = (c1a2 - c2a1)/(a1b2 - a2b1) */
        double a1, a2, b1, b2, c1, c2; // Line1: a1 x + b1 y + c1 =0; line2: a2 x + b2 y + c2=0
        a1 = linesInLeft[idLeft1][0].endPointY - linesInLeft[idLeft1][0].startPointY; // disY
        b1 = linesInLeft[idLeft1][0].startPointX - linesInLeft[idLeft1][0].endPointX; // -disX
        c1 = (0 - b1 * linesInLeft[idLeft1][0].startPointY) -
             a1 * linesInLeft[idLeft1][0].startPointX; // disX*sy - disY*sx
        length1 = linesInLeft[idLeft1][0].lineLength;

        a2 = linesInLeft[idLeft2][0].endPointY - linesInLeft[idLeft2][0].startPointY; // disY
        b2 = linesInLeft[idLeft2][0].startPointX - linesInLeft[idLeft2][0].endPointX; // -disX
        c2 = (0 - b2 * linesInLeft[idLeft2][0].startPointY) -
             a2 * linesInLeft[idLeft2][0].startPointX; // disX*sy - disY*sx
        length2 = linesInLeft[idLeft2][0].lineLength;

        a1b2_a2b1 = a1 * b2 - a2 * b1;
        if (abs(a1b2_a2b1) < 0.001)
        { // Two lines are almost parallel
          iRatio1L = numeric_limits<double>::infinity();
          iRatio2L = numeric_limits<double>::infinity();
        }
        else
        {
          double interSectionPointX = (c2 * b1 - c1 * b2) / a1b2_a2b1;
          double interSectionPointY = (c1 * a2 - c2 * a1) / a1b2_a2b1;
          // r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
          double disX = interSectionPointX - linesInLeft[idLeft1][0].startPointX;
          double disY = interSectionPointY - linesInLeft[idLeft1][0].startPointY;
          len = disY * a1 - disX * b1;
          iRatio1L = len / (length1 * length1);
          // r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
          disX = interSectionPointX - linesInLeft[idLeft2][0].startPointX;
          disY = interSectionPointY - linesInLeft[idLeft2][0].startPointY;
          len = disY * a2 - disX * b2;
          iRatio2L = len / (length2 * length2);
        }
        intersecRatioLeft.at<double>(idLeft1, idLeft2) = iRatio1L;
        intersecRatioLeft.at<double>(idLeft2, idLeft1) = iRatio2L; // Line order changed

        // Project the end points of line1 onto line2 and compute their distances to line2.
        double disS =
            abs(a2 * linesInLeft[idLeft1][0].startPointX + b2 * linesInLeft[idLeft1][0].startPointY + c2) / length2;
        double disE =
            abs(a2 * linesInLeft[idLeft1][0].endPointX + b2 * linesInLeft[idLeft1][0].endPointY + c2) / length2;
        pRatio1L = (disS + disE) / length1;
        projRatioLeft.at<double>(idLeft1, idLeft2) = pRatio1L;

        // Project the end points of line2 onto line1 and compute their distances to line1.
        disS = abs(a1 * linesInLeft[idLeft2][0].startPointX + b1 * linesInLeft[idLeft2][0].startPointY + c1) / length1;
        disE = abs(a1 * linesInLeft[idLeft2][0].endPointX + b1 * linesInLeft[idLeft2][0].endPointY + c1) / length1;
        pRatio2L = (disS + disE) / length2;
        projRatioLeft.at<double>(idLeft2, idLeft1) = pRatio2L;

        // Mark them as computed
        computedLeft.at(idLeft1 * numLineLeft + idLeft2) = true;
        computedLeft.at(idLeft2 * numLineLeft + idLeft1) = true;
      }
      else
      { // Read the information from the matrix.
        iRatio1L = intersecRatioLeft.at<double>(idLeft1, idLeft2);
        iRatio2L = intersecRatioLeft.at<double>(idLeft2, idLeft1);
        pRatio1L = projRatioLeft.at<double>(idLeft1, idLeft2);
        pRatio2L = projRatioLeft.at<double>(idLeft2, idLeft1);
      }

      if (not computedRight.at(idRight1 * numLineRight + idRight2))
      { // Have not been computed yet
        double a1, a2, b1, b2, c1, c2; // Line1: a1 x + b1 y + c1 =0; line2: a2 x + b2 y + c2=0
        a1 = linesInRight[idRight1][0].endPointY - linesInRight[idRight1][0].startPointY; // disY
        b1 = linesInRight[idRight1][0].startPointX - linesInRight[idRight1][0].endPointX; // -disX
        c1 = (0 - b1 * linesInRight[idRight1][0].startPointY) -
             a1 * linesInRight[idRight1][0].startPointX; // disX*sy - disY*sx
        length1 = linesInRight[idRight1][0].lineLength;

        a2 = linesInRight[idRight2][0].endPointY - linesInRight[idRight2][0].startPointY; // disY
        b2 = linesInRight[idRight2][0].startPointX - linesInRight[idRight2][0].endPointX; // -disX
        c2 = (0 - b2 * linesInRight[idRight2][0].startPointY) -
             a2 * linesInRight[idRight2][0].startPointX; // disX*sy - disY*sx
        length2 = linesInRight[idRight2][0].lineLength;

        a1b2_a2b1 = a1 * b2 - a2 * b1;
        if (abs(a1b2_a2b1) < 0.001)
        { // Two lines are almost parallel
          iRatio1R = numeric_limits<double>::infinity();
          iRatio2R = numeric_limits<double>::infinity();
        }
        else
        {
          double interSectionPointX = (c2 * b1 - c1 * b2) / a1b2_a2b1;
          double interSectionPointY = (c1 * a2 - c2 * a1) / a1b2_a2b1;
          // r1 = (s1I*s1e1)/(|s1e1|*|s1e1|)
          double disX = interSectionPointX - linesInRight[idRight1][0].startPointX;
          double disY = interSectionPointY - linesInRight[idRight1][0].startPointY;
          len = disY * a1 - disX * b1; // Because b1=-disX
          iRatio1R = len / (length1 * length1);
          // r2 = (s2I*s2e2)/(|s2e2|*|s2e2|)
          disX = interSectionPointX - linesInRight[idRight2][0].startPointX;
          disY = interSectionPointY - linesInRight[idRight2][0].startPointY;
          len = disY * a2 - disX * b2; // Because b2=-disX
          iRatio2R = len / (length2 * length2);
        }
        intersecRatioRight.at<double>(idRight1, idRight2) = iRatio1R;
        intersecRatioRight.at<double>(idRight2, idRight1) = iRatio2R;// line order changed

        // Project the end points of line1 onto line2 and compute their distances to line2.
        double disS =
            abs(a2 * linesInRight[idRight1][0].startPointX + b2 * linesInRight[idRight1][0].startPointY + c2) /
            length2;
        double disE =
            abs(a2 * linesInRight[idRight1][0].endPointX + b2 * linesInRight[idRight1][0].endPointY + c2) / length2;
        pRatio1R = (disS + disE) / length1;
        projRatioRight.at<double>(idRight1, idRight2) = pRatio1R;

        // Project the end points of line2 onto line1 and compute their distances to line1.
        disS = abs(a1 * linesInRight[idRight2][0].startPointX + b1 * linesInRight[idRight2][0].startPointY + c1) /
               length1;
        disE = abs(a1 * linesInRight[idRight2][0].endPointX + b1 * linesInRight[idRight2][0].endPointY + c1) / length1;
        pRatio2R = (disS + disE) / length2;
        projRatioRight.at<double>(idRight2, idRight1) = pRatio2R;

        // Mark them as computed
        computedRight.at(idRight1 * numLineRight + idRight2) = true;
        computedRight.at(idRight2 * numLineRight + idRight1) = true;
      }
      else
      { // Read these information from matrix
        iRatio1R = intersecRatioRight.at<double>(idRight1, idRight2);
        iRatio2R = intersecRatioRight.at<double>(idRight2, idRight1);
        pRatio1R = projRatioRight.at<double>(idRight1, idRight2);
        pRatio2R = projRatioRight.at<double>(idRight2, idRight1);
      }
      pRatioDif = MIN(abs(pRatio1L - pRatio1R), abs(pRatio2L - pRatio2R));
      if (pRatioDif > ProjectionRationDifThreshold)
      {
        continue;// The projection length ratio difference is too large;
      }
      if ((iRatio1L == numeric_limits<double>::infinity()) || (iRatio2L == numeric_limits<double>::infinity())
          || (iRatio1R == numeric_limits<double>::infinity()) || (iRatio2R == numeric_limits<double>::infinity()))
      {
        // Don't consider the intersection length ratio
        similarity = 4 - desDisMat.at<double>(idLeft1, idRight1) / parameters_.descriptorDifferenceThreshold
                     - desDisMat.at<double>(idLeft2, idRight2) / parameters_.descriptorDifferenceThreshold
                     - pRatioDif / ProjectionRationDifThreshold
                     - relativeAngleDif / parameters_.angleDifferenceThreshold;
        adjacenceVec[(2 * dim - j - 1) * j / 2 + i] = similarity;
        nnz++;
      }
      else
      {
        iRatioDif = MIN(abs(iRatio1L - iRatio1R), abs(iRatio2L - iRatio2R));
        if (iRatioDif > IntersectionRationDifThreshold)
        {
          continue;//the intersection length ratio difference is too large;
        }
        //now compute the similarity score between two line pairs.
        similarity = 5 - desDisMat.at<double>(idLeft1, idRight1) / parameters_.descriptorDifferenceThreshold
                     - desDisMat.at<double>(idLeft2, idRight2) / parameters_.descriptorDifferenceThreshold
                     - iRatioDif / IntersectionRationDifThreshold - pRatioDif / ProjectionRationDifThreshold
                     - relativeAngleDif / parameters_.angleDifferenceThreshold;
        adjacenceVec[(2 * dim - j - 1) * j / 2 + i] = similarity;
        nnz++;
      }
    }
  }

  // pointer to an array that stores the nonzero elements of Adjacency matrix.
  double *adjacenceMat = new double[nnz];
  // Array that stores the row indices of the non-zeros in adjacenceMat.
  int *irow = new int[nnz];
  // Array of indices pointing to the beginning of each column of adjacenceMat.
  int *pcol = new int[dim + 1];
  int idOfNNZ = 0; // The index counter of non-zero elements
  pcol[0] = 0;
  for (unsigned int j = 0; j < dim; j++)
  { // Column
    for (unsigned int i = j; i < dim; i++)
    { // Row
      unsigned int vectorIndex = (2 * dim - j - 1) * j / 2 + i;
      if (adjacenceVec[vectorIndex] != 0)
      {
        adjacenceMat[idOfNNZ] = adjacenceVec[vectorIndex];
        irow[idOfNNZ] = i;
        idOfNNZ++;
      }
    }
    pcol[j + 1] = idOfNNZ;
  }

#ifdef DEBUG_OUTPUT
  cout << "CCS Mat" << endl << "adjacenceMat= ";
  for (int i = 0; i < nnz; i++)
  {
    cout << adjacenceMat[i] << ", ";
  }
  cout << endl << "irow = ";
  for (int i = 0; i < nnz; i++)
  {
    cout << irow[i] << ", ";
  }
  cout << endl << "pcol = ";
  for (int i = 0; i < dim + 1; i++)
  {
    cout << pcol[i] << ", ";
  }
  cout << endl;
#endif // #ifdef DEBUG_OUTPUT

  /* Step 3: solve the principal eigenvector of the adjacency matrix using Arpack lib. */

  principalEigenVectorMAP_.clear();
  minOfPrincipalEigenVector_ = 0;

  ARluSymMatrix<double> arMatrix(dim, nnz, adjacenceMat, irow, pcol);

  if (dim <= 1)
  { // Cannot find eigenvectors if dimension too small
    delete[] adjacenceMat;
    delete[] irow;
    delete[] pcol;
    return;
  }

  // Defining what we need: the first eigenvector of arMatrix with largest magnitude.
  ARluSymStdEig<double> dprob(2, arMatrix, "LM");
  // Finding eigenvalues and eigenvectors.
  dprob.FindEigenvectors();

#ifdef DEBUG_OUTPUT
  cout << "Number of 'converged' eigenvalues  : " << dprob.ConvergedEigenvalues() << endl;

  cout << "eigenvalue is = " << dprob.Eigenvalue(0) << ", and " << dprob.Eigenvalue(1) << endl;
  if (dprob.EigenvectorsFound())
  {
    for (unsigned int j = 0; j < dim; j++)
    {
      cout << dprob.Eigenvector(1, j) << ", ";
    }
    cout << endl;
  }
#endif // #ifdef DEBUG_OUTPUT

  double meanEigenVec = 0;
  if (dprob.ConvergedEigenvalues() > 0)
  {
    unsigned int indexMaximumEigenvalue = (unsigned int) dprob.ConvergedEigenvalues() - 1;
    for (unsigned int j = 0; j < dim; j++)
    {
      double value = abs(dprob.Eigenvector(indexMaximumEigenvalue, j));
      meanEigenVec += value;
      principalEigenVectorMAP_.insert(std::make_pair(value, j));
    }
  }
  minOfPrincipalEigenVector_ = WeightOfMeanEigenVec * meanEigenVec / dim;

  delete[] adjacenceMat;
  delete[] irow;
  delete[] pcol;
}

void PairwiseLineMatching::MatchingResultFromPrincipalEigenvector_(ScaleLines &linesInLeft, ScaleLines &linesInRight,
                                                                   std::vector<unsigned int> &matchResult)
{
#ifdef DEBUG_OUTPUT
  //store eigenMap for debug
  ofstream resMap("eigenVec.txt", std::ios::out);
  Mat mat = Mat::zeros(linesInLeft.size(), linesInRight.size(), CV_64F);
  for (iter = principalEigenVectorMAP_.begin(); iter != principalEigenVectorMAP_.end(); iter++)
  {
    id = iter->second;
    resMap << nodesList_[id].leftLineID << "    " << nodesList_[id].rightLineID << "   " << iter->first << endl;
    mat.at<double>(nodesList_[id].leftLineID, nodesList_[id].rightLineID) = iter->first;
  }
  resMap.close();

  ofstream eigenMap("eigenMap.txt", ios::out);
  eigenMap << mat;
  eigenMap.close();
#endif // #ifdef DEBUG_OUTPUT

  std::vector<unsigned int> matchRet;
  double matchScore = 0;
  double relativeAngleLeft, relativeAngleRight; // The relative angle of each line pair
  double relativeAngleDif;

  // principalEigenVectorMAP_ is sorted in descending order.
  // Start from the highest element in the principal eigenvector.
  while (not principalEigenVectorMAP_.empty())
  {
    EigenMAP::iterator iter = principalEigenVectorMAP_.begin();

    // If the top element in the map has a small value, then there is no need to continue finding more
    // matching line pairs.
    if (iter->first < minOfPrincipalEigenVector_)
    {
      break;
    }
    unsigned int id = iter->second;
    unsigned int idLeft1 = nodesList_[id].leftLineID;
    unsigned int idRight1 = nodesList_[id].rightLineID;
    matchRet.push_back(idLeft1);
    matchRet.push_back(idRight1);
    matchScore += iter->first;
    iter = principalEigenVectorMAP_.erase(iter);

    // Remove all potential assignments in conflict with top matched line pair
    double xe_xsLeft = linesInLeft[idLeft1][0].endPointX - linesInLeft[idLeft1][0].startPointX;
    double ye_ysLeft = linesInLeft[idLeft1][0].endPointY - linesInLeft[idLeft1][0].startPointY;
    double xe_xsRight = linesInRight[idRight1][0].endPointX - linesInRight[idRight1][0].startPointX;
    double ye_ysRight = linesInRight[idRight1][0].endPointY - linesInRight[idRight1][0].startPointY;
    double coefLeft = (double) sqrt(xe_xsLeft * xe_xsLeft + ye_ysLeft * ye_ysLeft);
    double coefRight = (double) sqrt(xe_xsRight * xe_xsRight + ye_ysRight * ye_ysRight);
    while (iter != principalEigenVectorMAP_.end() and iter->first >= minOfPrincipalEigenVector_)
    {
      id = iter->second;
      unsigned int idLeft2 = nodesList_[id].leftLineID;
      unsigned int idRight2 = nodesList_[id].rightLineID;

      if ((idLeft1 == idLeft2) || (idRight1 == idRight2))
      { // Check one to one match condition
        iter = principalEigenVectorMAP_.erase(iter);
        continue;
      }

      // Check sidedness constraint: the middle point of line2 should lie on the same side of line1.
      // SideValue = (y-ys)*(xe-xs)-(x-xs)*(ye-ys);
      double pointX = 0.5 * (linesInLeft[idLeft2][0].startPointX + linesInLeft[idLeft2][0].endPointX);
      double pointY = 0.5 * (linesInLeft[idLeft2][0].startPointY + linesInLeft[idLeft2][0].endPointY);
      double sideValueL = (pointY - linesInLeft[idLeft1][0].startPointY) * xe_xsLeft
                          - (pointX - linesInLeft[idLeft1][0].startPointX) * ye_ysLeft;
      sideValueL = sideValueL / coefLeft;
      pointX = 0.5 * (linesInRight[idRight2][0].startPointX + linesInRight[idRight2][0].endPointX);
      pointY = 0.5 * (linesInRight[idRight2][0].startPointY + linesInRight[idRight2][0].endPointY);
      double sideValueR = (pointY - linesInRight[idRight1][0].startPointY) * xe_xsRight
                          - (pointX - linesInRight[idRight1][0].startPointX) * ye_ysRight;
      sideValueR = sideValueR / coefRight;

      if (sideValueL * sideValueR < 0 && abs(sideValueL) > 5 && abs(sideValueR) > 5)
      { // Have the different sign, conflict happens.
        iter = principalEigenVectorMAP_.erase(iter);
        continue;
      }

      // Check relative angle difference
      relativeAngleLeft = linesInLeft[idLeft1][0].direction - linesInLeft[idLeft2][0].direction;
      relativeAngleLeft = (relativeAngleLeft < PI) ? relativeAngleLeft : (relativeAngleLeft - TWO_PI);
      relativeAngleLeft = (relativeAngleLeft > (-PI)) ? relativeAngleLeft : (relativeAngleLeft + TWO_PI);
      relativeAngleRight = linesInRight[idRight1][0].direction - linesInRight[idRight2][0].direction;
      relativeAngleRight = (relativeAngleRight < PI) ? relativeAngleRight : (relativeAngleRight - TWO_PI);
      relativeAngleRight = (relativeAngleRight > (-PI)) ? relativeAngleRight : (relativeAngleRight + TWO_PI);
      relativeAngleDif = abs(relativeAngleLeft - relativeAngleRight);
      if ((TWO_PI - relativeAngleDif) > parameters_.angleDifferenceThreshold &&
          relativeAngleDif > parameters_.angleDifferenceThreshold)
      {
        principalEigenVectorMAP_.erase(iter++);
        continue; // The relative angle difference is too large;
      }
      iter++;
    }
  }
  matchResult = matchRet;
#ifdef DEBUG_OUTPUT
  cout << "matchRet.size" << matchRet.size() << ", minOfPrincipalEigenVector_= " << minOfPrincipalEigenVector_ << endl;
#endif // #ifdef DEBUG_OUTPUT
}

void
PairwiseLineMatching::PlotMatching(std::string path, std::vector<unsigned int> matching, cv::Mat imageLeft,
                                   ScaleLines &linesInLeft, cv::Mat imageRight,
                                   ScaleLines &linesInRight)
{
  srand(time(NULL));
  int lowest = 0, highest = 255;
  int range = (highest - lowest) + 1;

  for (unsigned int i = 0; i < linesInLeft.size(); i++)
  {
    unsigned int r = lowest + int(rand() % range);
    unsigned int g = lowest + int(rand() % range);
    unsigned int b = lowest + int(rand() % range);
    cv::Point startPoint = cv::Point(int(linesInLeft[i][0].startPointX), int(linesInLeft[i][0].startPointY));
    cv::Point endPoint = cv::Point(int(linesInLeft[i][0].endPointX), int(linesInLeft[i][0].endPointY));
    cv::line(imageLeft, startPoint, endPoint, cv::Scalar(r, g, b));
  }

  for (unsigned int i = 0; i < linesInRight.size(); i++)
  {
    unsigned int r = lowest + int(rand() % range);
    unsigned int g = lowest + int(rand() % range);
    unsigned int b = lowest + int(rand() % range);
    cv::Point startPoint = cv::Point(int(linesInRight[i][0].startPointX), int(linesInRight[i][0].startPointY));
    cv::Point endPoint = cv::Point(int(linesInRight[i][0].endPointX), int(linesInRight[i][0].endPointY));
    cv::line(imageRight, startPoint, endPoint, cv::Scalar(r, g, b));
  }

  std::vector<unsigned int> r1(matching.size() / 2), g1(matching.size() / 2), b1(
      matching.size() / 2); // The color of lines
  for (unsigned int pair = 0; pair < matching.size() / 2; pair++)
  {
    r1[pair] = lowest + int(rand() % range);
    g1[pair] = lowest + int(rand() % range);
    b1[pair] = 255 - r1[pair];
    double ww1 = 0.2 * (rand() % 5);
    double ww2 = 1 - ww1;
    char buf[10];
    sprintf(buf, "%d ", pair);
    int lineIDLeft = matching[2 * pair];
    int lineIDRight = matching[2 * pair + 1];
    cv::Point startPoint = cv::Point2i(int(linesInLeft[lineIDLeft][0].startPointX),
                                       int(linesInLeft[lineIDLeft][0].startPointY));
    cv::Point endPoint = cv::Point2i(int(linesInLeft[lineIDLeft][0].endPointX),
                                     int(linesInLeft[lineIDLeft][0].endPointY));
    line(imageLeft, startPoint, endPoint, CvScalar(r1[pair], g1[pair], b1[pair]), 3, CV_AA);
    startPoint = cv::Point2i(int(linesInRight[lineIDRight][0].startPointX),
                             int(linesInRight[lineIDRight][0].startPointY));
    endPoint = cv::Point2i(int(linesInRight[lineIDRight][0].endPointX), int(linesInRight[lineIDRight][0].endPointY));
    line(imageRight, startPoint, endPoint, CvScalar(r1[pair], g1[pair], b1[pair]), 3, CV_AA);
  }

  cv::Size sz = imageLeft.size();
  Mat result(sz.height, sz.width * 2, CV_8UC3);
  Mat result_left(result, cv::Rect(0, 0, sz.width, sz.height));
  imageLeft.copyTo(result_left);
  Mat result_right(result, cv::Rect(sz.width, 0, sz.width, sz.height));
  imageRight.copyTo(result_right);

  int imageWidth = imageLeft.cols;
  for (unsigned int pair = 0; pair < matching.size() / 2; pair++)
  {
    int lineIDLeft = matching[2 * pair];
    int lineIDRight = matching[2 * pair + 1];
    cv::Point startPoint = cvPoint(int(linesInLeft[lineIDLeft][0].startPointX),
                                   int(linesInLeft[lineIDLeft][0].startPointY));
    cv::Point endPoint = cvPoint(int(linesInRight[lineIDRight][0].startPointX + imageWidth),
                                 int(linesInRight[lineIDRight][0].startPointY));
    line(result, startPoint, endPoint, CvScalar(r1[pair], g1[pair], b1[pair]), 1, CV_AA);
  }

  imwrite(path, result);
}

PairwiseLineMatching::PairwiseLineMatching() : PairwiseLineMatching(PairwiseLineMatchingParameters())
{
}

PairwiseLineMatching::PairwiseLineMatching(PairwiseLineMatchingParameters parameters) : parameters_(parameters)
{
}

} // namespace lbd_descriptor
