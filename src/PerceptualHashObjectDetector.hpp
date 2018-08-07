/*
* object detector based on a perceptual hash.
*
* Copyright (c) 2018 Yaroslav Barabanov. Email: elektroyar@yandex.ru
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef PERCEPTUALHASHOBJECTDETECTOR_HPP_INCLUDED
#define PERCEPTUALHASHOBJECTDETECTOR_HPP_INCLUDED

#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

namespace PerceptualHashObjectDetector {


    /**@brief ����� �������� ����������� �� 25 ��� ����
        ������� �������� �� ����� ������������� ����������� � ������ ����� ��������, ��� ��� � ������ ������� ��������
        ����������� (������ ���� �������� ������������� ���� � ���������� ������� ������� ��������������� � �������� dataHash � dataMean).
        @param[in] inputImage ������������ ����������� ���� CV_32FC1
        @param[in] dataHash ������ �� 33554432 ���������, ��� ����� �������� - 25 ������ ���
        @param[in] dataMean ������ �� 256 ��������� � ������ ���������� ������ ������� �������
        @param[in] scaleMin ����������� �������
        @param[in] scaleMax ������������ �������
        @param[in] scaleStep ��� ��������
        @param[in] stepValue ������� �������� ��� ���� � ������ �� ������ ��� ������ ��������������� �����
        @param[out] rect ������ ��������������� �����
        @param[out] rectHash ������ ����� ��������������� �����, ����� ������������ ��� �������������
    */
    void searhIntegralImage(cv::Mat& inputImage, unsigned char* dataHash, unsigned char* dataMean,
            double scaleMin, double scaleMax, double scaleStep, double stepValue,
            int w, int h, std::vector<cv::Rect>& rect, std::vector<unsigned long>& rectHash);

    /**@brief ������������ ��� ������� ����������� � �������� ������� �������
        ������� ���������� ��� ������� �����������. ������� ��������� ������������ �����������
        CV_32FC1 ��� ����� ����������� CV_8UC1 ��� ������� ����������� CV_8UC3.
        @param[in] inputImage �������� �����������
        @param[in] rect ��������������� �����
        @param[in] wHash ������ ��� ������� ������������� ����
        @param[in] hHash ������ ��� ������� ������������� ����
        @param[in,out] outMean ������� ������� ������� �����������
        @return ������������ ��� ������� �����������, �������� �� 32 ���
    */
    unsigned long getHash32(cv::Mat& inputImage, cv::Rect& rect, int wHash, int hHash, unsigned char* outMean);

    /**@brief ������������ ��� ������� �����������
        ������� ���������� ��� ������� �����������. ������� ��������� ������������ �����������
        CV_32FC1 ��� ����� ����������� CV_8UC1 ��� ������� ����������� CV_8UC3.
        @param[in] inputImage �������� �����������
        @param[in] rect ��������������� �����
        @param[in] wHash ������ ��� ������� ������������� ����
        @param[in] hHash ������ ��� ������� ������������� ����
        @return ������������ ��� ������� �����������, �������� �� 32 ���
    */
    unsigned long getHash32(cv::Mat& inputImage, cv::Rect& rect, int wHash, int hHash);

    /**@brief �������� ����������� ������
        @param[in] hashImage ��� ������� �����������
        @param[in] outArray ������, ���������� ����������
        @param[in] hashData ����
        @param[in] hammingDistance ���������� ��������
        @param[in] maxBit ������������ ����� ���
    */
    void getNoise32(unsigned long hashImage, unsigned char* outArray, unsigned char hashData, unsigned char hammingDistance, unsigned char maxBit);

    void getMeanNoise(unsigned char* meanArray, unsigned char mean, unsigned char noise);

    void showHash32(unsigned long hashImage, cv::Mat& image, int hashW, int hashH, int w, int h, bool isRGB);

}
#endif // PERCEPTUALHASHOBJECTDETECTOR_HPP_INCLUDED
