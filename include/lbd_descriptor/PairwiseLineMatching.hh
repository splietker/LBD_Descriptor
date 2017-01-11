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

#ifndef PAIRWISELINEMATCHING_HH_
#define PAIRWISELINEMATCHING_HH_

#include <map>
#include <string>
#include <opencv2/core/mat.hpp>

#include "LineDescriptor.hh"


namespace lbd_descriptor
{

struct PairwiseLineMatchingParameters
{
  PairwiseLineMatchingParameters() :
      globalRotationAngleEnabled(true),
      globalRotationHistogramResolutionScale(10),
      globalRotationAcceptableHistogramAngle(0.49),
      globalRotationAcceptableLengthDifference(0.4),
      descriptorDifferenceThreshold(0.35),
      lengthDifferenceThreshold(4),
      angleDifferenceThreshold(M_PI / 4)
  {}

  /**
   * Flag determining whether to compute the global rotation angle between image pairs.
   * Default value is true.
   */
  bool globalRotationAngleEnabled;

  /**
   * The resolution scale of theta histogram, used when compute angle histogram of lines in the image.
   * Default value is 10.
   */
  double globalRotationHistogramResolutionScale;

  /**
   * Threshold for accepting the histogram's rotation angle. Some image pairs don't have a stable global rotation
   * angle, e.g. the image pair of the wide baseline non planar scene.
   * Default value is 0.49.
   */
  double globalRotationAcceptableHistogramAngle;

  /**
   * Threshold for accepting the length vector in the rotation computation.
   * Default value is 0.4.
   */
  double globalRotationAcceptableLengthDifference;

  /**
   * Maximum difference between descriptors of two lines to be considered as possible matching pair.
   * Default value is 0.35.
   */
  double descriptorDifferenceThreshold;

  /**
   * Maximum length difference of two lines to be considered as possible matching pair.
   * Default value is 4.
   */
  double lengthDifferenceThreshold;

  /**
   * Maximum relative angle between two lines to be considered as possible matching pair.
   * Default value is PI/4.
   */
  double angleDifferenceThreshold;
};

/**
 * Each node in the graph is a possible line matching pair in the left and right image.
 */
struct LBDNode
{
  unsigned int leftLineID; // The index of line in the left image.
  unsigned int rightLineID; // The index of line in the right image.
};

// Specifies a vector of nodes.
typedef std::vector<LBDNode> Nodes_list;

struct CompareL
{
  bool operator()(const double &lhs, const double &rhs) const
  { return lhs > rhs; }
};

typedef std::multimap<double, unsigned int, CompareL> EigenMAP;

struct CompareS
{
  bool operator()(const double &lhs, const double &rhs) const
  { return lhs < rhs; }
};

typedef std::multimap<double, unsigned int, CompareS> DISMAP;

class PairwiseLineMatching
{
public:
  PairwiseLineMatching();

  PairwiseLineMatching(PairwiseLineMatchingParameters parameters);

  void LineMatching(ScaleLines &linesInLeft, ScaleLines &linesInRight,
                    std::vector<unsigned int> &matchResult);

  static void PlotMatching(std::string path, std::vector<unsigned int> matching, cv::Mat imageLeft, ScaleLines &linesInLeft, cv::Mat imageRight,
                           ScaleLines &linesInRight);

private:
  /**
   * Compute the approximate global rotation angle between image pair(i.e. the left and right images).
   *
   * As shown in Bin Fan's work "Robust line matching through line-point invariants", this approximate
   * global rotation angle can greatly prune the spurious line correspondences. This is the idea of their
   * fast matching version. Nevertheless, the approaches to estimate the approximate global rotation angle
   * are different. Their is based on the rotation information included in the matched point feature(such as SIFT)
   * while ours is computed from angle histograms of lines in images. Our approach also detect whether there is an
   * appropriate global rotation angle between image pair.
   * step 1: Get the angle histograms of detected lines in the left and right images, respectively;
   * step 2: Search the shift angle between two histograms to minimize their difference. Take this shift angle as
   *         approximate global rotation angle between image pair.
   * input:  detected lines in the left and right images
   * return: the global rotation angle
   */
  double GlobalRotationOfImagePair_(ScaleLines &linesInLeft, ScaleLines &linesInRight);

  /**
   * Build the symmetric non-negative adjacency matrix M, whose nodes are the potential assignments a = (i_l, j_r)
   * and whose weights on edges measure the agreements between pairs of potential assignments. That is where the pairwise
   * constraints are applied(c.f. A spectral technique for correspondence problems using pairwise constraints, M.Leordeanu).
   */
  void BuildAdjacencyMatrix_(ScaleLines &linesInLeft, ScaleLines &linesInRight);

  /**
   * Get the final matching from the principal eigenvector.
   */
  void MatchingResultFromPrincipalEigenvector_(ScaleLines &linesInLeft, ScaleLines &linesInRight,
                                               std::vector<unsigned int> &matchResult);

  /**
   * Parameters for PairwiseLineMatching.
   */
  PairwiseLineMatchingParameters parameters_;

  /**
   * The approximate global rotation angle between image pairs.
   */
  double globalRotationAngle_;

  /**
   * Construct a map to store the principal eigenvector and its index.
   * Each pair in the map is in this form (eigenvalue, index);
   * Note that, we use eigenvalue as key in the map and index as their value.
   * This is because the map need be sorted by the eigenvalue rather than index
   * for our purpose.
   */
  EigenMAP principalEigenVectorMAP_;

  /**
   * Save all the possible matched line pairs.
   */
  Nodes_list nodesList_;

  /**
   * The acceptable minimal value in the principal eigen vector.
   */
  double minOfPrincipalEigenVector_;
};

} // namespace lbd_descriptor

#endif /* PAIRWISELINEMATCHING_HH_ */
