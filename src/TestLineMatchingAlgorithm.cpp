/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2011-2012, Lilian Zhang, all rights reserved.
Copyright (C) 2013, Manuele Tamburrano, Stefano Fabri, all rights reserved.
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


#include <math.h>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>
#include <set>

#include "lbd_descriptor/LineDescriptor.hh"
#include "lbd_descriptor/PairwiseLineMatching.hh"

using namespace cv;
using namespace std;
using namespace lbd_descriptor;

void printUsage(int argc, char **argv)
{
  cout << "Usage: " << argv[0] << "  image1.png" << "  image2.png" << endl;
}

void KeyLinesToScaleLines(vector<line_descriptor::KeyLine> &keyLines, ScaleLines &scaleLines, Mat &descriptors)
{
  map<int, int> numOctavesMapping;
  for (line_descriptor::KeyLine kl: keyLines)
  {
    numOctavesMapping[kl.class_id] += 1;
  }
  scaleLines.clear();
  scaleLines.resize(numOctavesMapping.size());
  unsigned int descriptorIndex = 0;
  for (line_descriptor::KeyLine kl: keyLines)
  {
    OctaveSingleLine osl;

    float *row = descriptors.ptr<float>(descriptorIndex);
    copy(row, row + descriptors.cols, back_inserter(osl.descriptor));

    osl.startPointX = kl.startPointX;
    osl.startPointY = kl.startPointY;
    osl.endPointX = kl.endPointX;
    osl.endPointY = kl.endPointY;
    osl.sPointInOctaveX = kl.sPointInOctaveX;
    osl.sPointInOctaveY = kl.sPointInOctaveY;
    osl.ePointInOctaveX = kl.ePointInOctaveX;
    osl.ePointInOctaveY = kl.ePointInOctaveY;
    osl.lineLength = kl.lineLength;
    osl.numOfPixels = kl.numOfPixels;
    osl.salience = kl.response;
    osl.direction = kl.angle;
    osl.octaveCount = kl.octave;

    scaleLines.at(kl.class_id).push_back(osl);

    descriptorIndex += 1;
  }
}

int main(int argc, char **argv)
{
  int ret = -1;
  if (argc < 3)
  {
    printUsage(argc, argv);
    return ret;
  }
  //load first image from file
  std::string imageName1(argv[1]);
  cv::Mat leftImage;
  leftImage = imread(imageName1, cv::IMREAD_GRAYSCALE);   // Read the file
  if (!leftImage.data)                              // Check for invalid input
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  //load second image from file
  std::string imageName2(argv[2]);
  cv::Mat rightImage;
  rightImage = imread(imageName2, cv::IMREAD_GRAYSCALE);   // Read the file
  if (!rightImage.data)                              // Check for invalid input
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  //initial variables
  cv::Mat leftColorImage(leftImage.size(), CV_8UC3);
  cv::Mat rightColorImage(rightImage.size(), CV_8UC3);

  cvtColor(leftImage, leftColorImage, cv::COLOR_GRAY2RGB);
  cvtColor(rightImage, rightColorImage, cv::COLOR_GRAY2RGB);

  PairwiseLineMatching lineMatch;

  ///////////####################################################################
  ///////////####################################################################

  line_descriptor::BinaryDescriptor::Params params;
  params.numOfOctave_ = 2;
  params.widthOfBand_ = 7;
  //params.reductionRatio = sqrt(2);
  Ptr<line_descriptor::BinaryDescriptor> binary_descriptor = line_descriptor::BinaryDescriptor::createBinaryDescriptor(
      params);

  vector<line_descriptor::KeyLine> keyLinesLeft, keyLinesRight;
  Mat descriptorsLeft, descriptorsRight;

  Mat binary_detector_mask = Mat::ones(Size(leftImage.cols, leftImage.rows), CV_8UC1);
  (*binary_descriptor)(leftImage, binary_detector_mask, keyLinesLeft, descriptorsLeft, false, true);
  (*binary_descriptor)(rightImage, binary_detector_mask, keyLinesRight, descriptorsRight, false, true);

  ScaleLines sll, slr;
  KeyLinesToScaleLines(keyLinesLeft, sll, descriptorsLeft);
  KeyLinesToScaleLines(keyLinesRight, slr, descriptorsRight);

  std::vector<unsigned int> matchResultBD;
  lineMatch.LineMatching(sll, slr, matchResultBD);
  lineMatch.PlotMatching("LBDSG_BD.png", matchResultBD, leftColorImage, sll, rightColorImage, slr);

  ///////////####################################################################
  ///////////####################################################################

  //extract lines, compute their descriptors and match lines
  LineDescriptor lineDesc;

  ScaleLines linesInLeft;
  ScaleLines linesInRight;
  std::vector<unsigned int> matchResult;

  lineDesc.GetKeyLines(leftImage, linesInLeft);
  lineDesc.GetKeyLines(rightImage, linesInRight);

	lineDesc.ComputeDescriptors(linesInLeft);
	lineDesc.ComputeDescriptors(linesInRight);

 	lineMatch.LineMatching(linesInLeft,linesInRight,matchResult);
	lineMatch.PlotMatching("LBDSG_LBD.png", matchResult, leftColorImage, linesInLeft, rightColorImage, linesInRight);
}
