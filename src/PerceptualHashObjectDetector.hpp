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


    /**@brief Поиск участкая изображения по 25 бит хэшу
        Функция проходит по всему интегральному изображению в поиске таких участков, чей хэш и средяя яркость является
        допустимыми (массив всех значений перцептивного хэша и допустимая средняя яркость устанавливаются в массивах dataHash и dataMean).
        @param[in] inputImage интегральное изображение типа CV_32FC1
        @param[in] dataHash массив из 33554432 элементов, где номер элемента - 25 битный хэш
        @param[in] dataMean массив из 256 элементов с флагом разрешения данной средней яркости
        @param[in] scaleMin минимальный масштаб
        @param[in] scaleMax максимальный масштаб
        @param[in] scaleStep шаг масштаба
        @param[in] stepValue процент пикселей для шага в циклах от ширины или высоты ограничительной рамки
        @param[out] rect массив ограничительных рамок
        @param[out] rectHash массив хэшей ограничительных рамок, можно использовать для мультитрекера
    */
    void searhIntegralImage(cv::Mat& inputImage, unsigned char* dataHash, unsigned char* dataMean,
            double scaleMin, double scaleMax, double scaleStep, double stepValue,
            int w, int h, std::vector<cv::Rect>& rect, std::vector<unsigned long>& rectHash);

    /**@brief Перцептивный хэш участка изображения с массивом средней яркости
        Функция возвращает хэш участка изображения. Функция принимает интегральное изображение
        CV_32FC1 или серое изображение CV_8UC1 или цветное изображение CV_8UC3.
        @param[in] inputImage входящее изображение
        @param[in] rect ограничительная рамка
        @param[in] wHash ширина дря расчета перцептивного хэша
        @param[in] hHash высота для расчета перцептивного хэша
        @param[in,out] outMean средняя яркость участка изображения
        @return перцептивный хэш участка изображения, размером до 32 бит
    */
    unsigned long getHash32(cv::Mat& inputImage, cv::Rect& rect, int wHash, int hHash, unsigned char* outMean);

    /**@brief Перцептивный хэш участка изображения
        Функция возвращает хэш участка изображения. Функция принимает интегральное изображение
        CV_32FC1 или серое изображение CV_8UC1 или цветное изображение CV_8UC3.
        @param[in] inputImage входящее изображение
        @param[in] rect ограничительная рамка
        @param[in] wHash ширина для расчета перцептивного хэша
        @param[in] hHash высота для расчета перцептивного хэша
        @return перцептивный хэш участка изображения, размером до 32 бит
    */
    unsigned long getHash32(cv::Mat& inputImage, cv::Rect& rect, int wHash, int hHash);

    /**@brief Получить зашумленные данные
        @param[in] hashImage хэш участка изображения
        @param[in] outArray массив, подлежащий зашумлению
        @param[in] hashData флаг
        @param[in] hammingDistance расстояние хэмминга
        @param[in] maxBit максимальное число бит
    */
    void getNoise32(unsigned long hashImage, unsigned char* outArray, unsigned char hashData, unsigned char hammingDistance, unsigned char maxBit);

    void getMeanNoise(unsigned char* meanArray, unsigned char mean, unsigned char noise);

    void showHash32(unsigned long hashImage, cv::Mat& image, int hashW, int hashH, int w, int h, bool isRGB);

}
#endif // PERCEPTUALHASHOBJECTDETECTOR_HPP_INCLUDED
