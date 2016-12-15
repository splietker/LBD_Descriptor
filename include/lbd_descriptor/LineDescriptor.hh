/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2011-2012, Lilian Zhang, all rights reserved.
Copyright (C) 2013, Manuele Tamburrano, Stefano Fabri, all rights reserved.
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

#ifndef LINEDESCRIPTOR_HH_
#define LINEDESCRIPTOR_HH_

#include <map>

#include "EDLineDetector.hh"
#include "LineStructure.hh"


namespace lbd_descriptor
{

struct OctaveLine
{
  /**
   * The octave which this line is detected.
   */
  unsigned int octaveCount;

  /**
   * The line ID in that octave image.
   */
  unsigned int lineIDInOctave;

  /**
   * The line ID in Scale line vector.
   */
  unsigned int lineIDInScaleLineVec;

  /**
   * The length of line in original image scale.
   */
  float lineLength;
};

struct LineDescriptorParameters
{
  LineDescriptorParameters() :
      ksize(5),
      numOfOctave(5),
      numOfBand(9),
      widthOfBand(7),
      lowestThreshold(0.35),
      NNDRThreshold(0.6)
  {}

  /**
   * The size of Gaussian kernel: ksize X ksize.
   * Default value is 5.
   */
  int ksize;

  /**
   * The number of image octave.
   * Default value is 5.
   */
  unsigned int numOfOctave;

  /**
   * The number of band used to compute line descriptor.
   * Default value is 9.
   */
  unsigned int numOfBand;

  /**
   * The width of band.
   * Default value is 7.
   */
  unsigned int widthOfBand;

  /**
   * Global threshold for line descriptor distance.
   * 2 is used to show recall ratio;  0.2 is used to show scale space results, 0.35 is used when verify geometric
   * constraints. Default value is 0.35
   */
  float lowestThreshold;

  /**
   * The NNDR threshold for line descriptor distance.
   * Default value is 0.6.
   */
  float NNDRThreshold;

  /**
   * EDLine parameters.
   */
  EDLineParam edLineParam;
};

/**
 * This class is used to generate the line descriptors from multi-scale images
 */
class LineDescriptor
{
public:
  LineDescriptor();

  LineDescriptor(LineDescriptorParameters parameters);

  ~LineDescriptor();

  enum
  {
    NearestNeighbor = 0, // The nearest neighbor is taken as matching
    NNDR = 1 // Nearest/next ratio
  };

  /**
   * This function is used to detect lines from multi-scale images.
   */
  int GetKeyLines(cv::Mat &image, ScaleLines &keyLines);

  /**
   * Compute the line descriptor of input line set. This function should be called
   * after OctaveKeyLines() function.
   */
  int ComputeDescriptors(ScaleLines &keyLines);

  /**
   * Match line by their descriptors.
   * The function will use opencv FlannBasedMatcher to match lines.
   */
  int MatchLineByDescriptor(ScaleLines &keyLinesLeft, ScaleLines &keyLinesRight,
                            std::vector<short> &matchLeft, std::vector<short> &matchRight,
                            int criteria = NNDR);

private:
  static void sample(float *igray, float *ogray, float factor, int width, int height)
  {

    int swidth = (int) ((float) width / factor);
    int sheight = (int) ((float) height / factor);

    for (int j = 0; j < sheight; j++)
      for (int i = 0; i < swidth; i++)
        ogray[j * swidth + i] = igray[(int) ((float) j * factor) * width + (int) ((float) i * factor)];

  }

  static void sampleUchar(uchar *igray, uchar *ogray, float factor, int width, int height)
  {

    int swidth = (int) ((float) width / factor);
    int sheight = (int) ((float) height / factor);

    for (int j = 0; j < sheight; j++)
      for (int i = 0; i < swidth; i++)
        ogray[j * swidth + i] = igray[(int) ((float) j * factor) * width + (int) ((float) i * factor)];

  }

  /**
   * For each octave of image, we define an EDLineDetector, because we can get gradient images (dxImg, dyImg, gImg)
   * from the EDLineDetector class without extra computation cost. Another reason is that, if we use
   * a single EDLineDetector to detect lines in different octave of images, then we need to allocate and release
   * memory for gradient images (dxImg, dyImg, gImg) repeatedly for their varying size
   */
  std::vector<EDLineDetector *> edLineVec_;

  /**
   * Parameters for LineDescriptor.
   */
  LineDescriptorParameters parameters_;

  /**
   * The local gaussian coefficient apply to the orthogonal line direction within each band.
   */
  std::vector<float> gaussCoefL_;

  /**
   * The global gaussian coefficient applied to each Row within the line support region.
   */
  std::vector<float> gaussCoefG_;
};

} // namespace lbd_descriptor

#endif /* LINEDESCRIPTOR_HH_ */
