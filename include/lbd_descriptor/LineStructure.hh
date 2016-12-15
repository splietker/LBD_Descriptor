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

#ifndef LINESTRUCTURE_HH_
#define LINESTRUCTURE_HH_

#include <vector>


namespace lbd_descriptor
{

struct OctaveSingleLine
{
	/*endPoints, the coordinate origin is the top-left corner of the original image.
	 *startPointX = sPointInOctaveX * (factor)^octaveCount;	*/
	float startPointX;
	float startPointY;
	float endPointX;
	float endPointY;
	//endPoints, the coordinate origin is the top-left corner of the octave image.
	float sPointInOctaveX;
	float sPointInOctaveY;
	float ePointInOctaveX;
	float ePointInOctaveY;
	//direction of a line, the angle between positive line direction (dark side is in the left) and positive X axis.
	float direction;
	//the summation of gradient magnitudes of pixels on lines
	float salience;
	//the length of line
	float lineLength;
	//number of pixels
	unsigned int numOfPixels;
	//the octave which this line is detected
	unsigned int octaveCount;
	//the decriptor of line
	std::vector<float> descriptor;
};

/**
 * Specifies a vector of lines.
 */
typedef std::vector<OctaveSingleLine> LinesVec;

/**
 * Each element in ScaleLines is a vector of lines which corresponds the same line detected in different octave images.
 */
typedef std::vector<LinesVec> ScaleLines;

} // namespace lbd_descriptor

#endif /* LINESTRUCTURE_HH_ */
