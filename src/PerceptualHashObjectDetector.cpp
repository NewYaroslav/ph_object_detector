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

#include "PerceptualHashObjectDetector.hpp"
#include <thread>

namespace PerceptualHashObjectDetector {
    static const unsigned long tableHashBit[32] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,
        8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,
        33554432,67108864,134217728,268435456,536870912,1073741824,2147483648};

    // функци€ принимает на вход интегральное изображение
    static void threadSearhIntegralImage(cv::Mat& inputImage, unsigned char* dataHash, unsigned char* dataMean,
        double scaleMin, double scaleMax, double scaleStep, double stepValue, double stepOffset,
        int w, int h,
        std::vector<cv::Rect>& rect, std::vector<unsigned long>& rectHash) {
        rect.clear();
        rectHash.clear();
        const int hashStandartWH = 5;
        double scale = scaleMin; // минимальный масштаб
        int rectW = w * scale;
        int rectH = h * scale;
        int const imageWidth(inputImage.cols), imageHeight(inputImage.rows);
        unsigned long hashData; // хэш
        float mean; // средн€€ €ркость
        float data; // €ркость области в сетке 5 х 5
        float numPixRect; // площадь ограничительой рамки, нужна дл€ нахождени€ mean
        //short x2,y3; // угловые точки ограничительной рамки
        short rectX2, rectY3; // угловые точки области в сетке 5 х 5
        int i = 0; // инкремент дл€ данных хэша

        while(1) {
            if (scale >= scaleMax)
                break;
            numPixRect = rectW * rectH;
            short incX = rectW / hashStandartWH; // прирост по X
            short incY = rectH / hashStandartWH; // прирост по Y
            float numPixRect5x5 = incX * incY; // площадь ограничительой рамки, нужна дл€ нахождени€ mean
            int incYx5 = incY * hashStandartWH;
            int incXx5 = incX * hashStandartWH;

            int incAll = stepValue * std::min(rectW, rectH);
            int offsetX = stepOffset * incAll;
            //printf("rectW =  %d rectH = %d scale = %f incAll = %d\n",rectW,rectH,scale,incAll);
            if (incAll <= 1) {
                incAll = 1;
                offsetX = 0;
            }

            for(int y(imageHeight - incYx5 - 1); y >= 0; y-=incAll) {
                int maxY = y + incYx5;
                unsigned long *const lineH( inputImage.ptr<unsigned long>(y) );
                unsigned long *const lineL( inputImage.ptr<unsigned long>(maxY) );
                for(int x(imageWidth - incXx5 - offsetX - 1); x >= 0 ; x-=incAll) {
                    ///======================================================
                    int x2 = x + incXx5;

                    mean = (float)lineH[x] + (float)lineL[x2]
                          -(float)lineH[x2] -  (float)lineL[x];
                    mean = mean / numPixRect;

                    if (!dataMean[(unsigned char)mean]) {
                        continue; // если средн€€ €ркость участка€ изображени€ €вл€етс€ запрещенной, то пропустим
                    }

                    int const maxX = x + incXx5;
                    // считаем хэш участка€ изображени€
                    i = 0;
                    hashData = 0;
                    for (short _y = y; _y < maxY; _y += incY) {
                        rectY3 = _y + incY;
                        unsigned long *const _lineH( inputImage.ptr<unsigned long>(_y) );
                        unsigned long *const _lineL( inputImage.ptr<unsigned long>(rectY3) );
                        for (short _x = x; _x < maxX; _x += incX) {
                            rectX2 = _x + incX;
                            data = (float)_lineH[_x] + (float)_lineL[rectX2]
                                 - (float)_lineH[rectX2] - (float)_lineL[_x];
                            data = data / numPixRect5x5;
                            hashData |= (data > mean) ? tableHashBit[i] : 0;
                            i++;
                        }
                    }
                    // если хэш существует
                    if (dataHash[hashData]) {
                        cv::Rect rectData(x > 0 ? x - 1 : 0, y > 0 ? y - 1 : 0, rectW, rectH);
                        // загрузим рамку и флаг рамки
                        rect.push_back(rectData);
                        rectHash.push_back(hashData);
                    }
                }
            }
            // увеличим масштаб
            scale += scaleStep;
            rectW = ceil(scale * w);
            rectH = ceil(scale * h);
            //printf("rectW =  %d rectH = %d scale = %f\n",rectW,rectH,scale);
        }
    }

    // функци€ принимает на вход интегральное изображение
    void searhIntegralImage(cv::Mat& inputImage, unsigned char* dataHash, unsigned char* dataMean,
        double scaleMin, double scaleMax, double scaleStep, double stepValue,
        int w, int h,
        std::vector<cv::Rect>& rect, std::vector<unsigned long>& rectHash) {
        rect.clear();
        rectHash.clear();
        unsigned int n = std::thread::hardware_concurrency();
        if (n < 2) {
            // запускаем один поток с нулевым сдвигом поискового окна
            threadSearhIntegralImage(inputImage, dataHash, dataMean,
                                     scaleMin, scaleMax, scaleStep, stepValue, 0.0,
                                     w, h,
                                     rect, rectHash);
        } else {
            std::thread threadArray[n]; // создаем массив потоков
            std::vector<std::vector<cv::Rect>> rectArray(n);
            std::vector<std::vector<unsigned long>> rectHashArray(n);
            double newStepValue = stepValue * n;
            double stepOffset[n];
            stepOffset[0] = 0.0;
            for(unsigned int th = 1; th < n; th++) {
                stepOffset[th] = stepOffset[th - 1] + (1.0 / (double) n);
            }
            for(unsigned int th = 0; th < n; th++) {
                threadArray[th] = std::thread(&threadSearhIntegralImage,std::ref(inputImage), std::ref(dataHash), std::ref(dataMean),
                    scaleMin, scaleMax, scaleStep, newStepValue, stepOffset[th], w, h,
                    std::ref(rectArray[th]), std::ref(rectHashArray[th]));
            }
            for(unsigned int th = 0; th < n; th++) {
                threadArray[th].join();
            }
            for(unsigned int th = 0; th < n; th++) {
                if (rectArray[th].size() > 0) {
                    rect.insert(rect.end(), rectArray[th].begin(), rectArray[th].end());
                    rectHash.insert(rectHash.end(), rectHashArray[th].begin(), rectHashArray[th].end());
                }
            } // for
        } // if
    } // searhIntegralImage


    // функци€ возвращает хэш интегрального изображени€ или обычного изображени€
    unsigned long getHash32(cv::Mat& inputImage, cv::Rect& rect, int wHash, int hHash) {
        if ((wHash * hHash) > 32) {
            printf("Error getHashImage: hash size is larger than 32 bits.");
            return 0;
        }
        cv::Mat inputIntegralImage;
        if (inputImage.type() != CV_32FC1) {
            if (inputImage.type() == CV_8UC3) {
                cv::Mat inputGrayImage;
                cv::Mat imageIntegral(inputImage.cols + 1,inputImage.rows + 1,CV_32FC1);
                cv::cvtColor( inputImage, inputGrayImage, CV_BGR2GRAY );
                cv::integral(inputGrayImage, imageIntegral);
                inputIntegralImage = imageIntegral.clone();
                imageIntegral.release();
                inputGrayImage.release();
            } else
            if (inputImage.type() == CV_8UC1) {
                cv::Mat imageIntegral(inputImage.cols + 1,inputImage.rows + 1,CV_32FC1);
                cv::integral(inputImage, imageIntegral);
                inputIntegralImage = imageIntegral.clone();
                imageIntegral.release();
            } else {
                printf("Error getHashImage: The input video format inappropriate.");
                return 0;
            }
        } else {
            inputIntegralImage = inputImage.clone();
        }

        short x2,y3;
        //int const imageWidth(image.cols), imageHeight(image.rows);
        // точки пр€моугольника
        x2 = rect.x + rect.width - 1;
        y3 = rect.y + rect.height - 1;

        unsigned long *const lineH( inputIntegralImage.ptr<unsigned long>(rect.y - 1) );
        unsigned long *const lineL( inputIntegralImage.ptr<unsigned long>(y3) );
        float mean = (float)lineH[rect.x - 1] + (float)lineL[x2]
                    -(float)lineH[x2] -  (float)lineL[rect.x - 1];
        float kMean = rect.width * rect.height;
        mean = mean / kMean;

        //dataMean[(unsigned char)mean] = 1;
        float data;
        short incX = rect.width / wHash;
        short incY = rect.height / hHash;
        if (incX == 0 || incY == 0 ) {
            printf("Error getHashImage: width or height of the bounding box is too small.");
            return 0;
        }
        float kMeanShRect = incX * incY;
        short rectX2, rectY3;
        unsigned long hashImage = 0;
        int i = 0;

        int const maxY = rect.y + incY * hHash - 1;
        int const maxX = rect.x + incX * wHash - 1;

        for (short y = rect.y - 1; y < maxY; y += incY) {
            rectY3 = y + incY;
            unsigned long *const lineH( inputIntegralImage.ptr<unsigned long>(y) );
            unsigned long *const lineL( inputIntegralImage.ptr<unsigned long>(rectY3) );
            for (short x = rect.x - 1; x < maxX; x += incX) {
                rectX2 = x + incX;
                data = (float)lineH[x] + (float)lineL[rectX2]
                     - (float)lineH[rectX2] - (float)lineL[x];
                data = data / kMeanShRect;
                hashImage |= (data > mean) ? tableHashBit[i] : 0;
                i++;
            }
        }
        inputIntegralImage.release();
        return hashImage;
    }

    // функци€ возвращает хэш интегрального изображени€ или обычного изображени€
    unsigned long getHash32(cv::Mat& inputImage, cv::Rect& rect, int wHash, int hHash, unsigned char* outMean) {
        if ((wHash * hHash) > 32) {
            printf("Error getHashImage: hash size is larger than 32 bits.");
            return 0;
        }
        cv::Mat inputIntegralImage;
        if (inputImage.type() != CV_32FC1) {
            if (inputImage.type() == CV_8UC3) {
                cv::Mat inputGrayImage;
                cv::Mat imageIntegral(inputImage.cols + 1,inputImage.rows + 1,CV_32FC1);
                cv::cvtColor( inputImage, inputGrayImage, CV_BGR2GRAY );
                cv::integral(inputGrayImage, imageIntegral);
                inputIntegralImage = imageIntegral.clone();
                imageIntegral.release();
                inputGrayImage.release();
            } else
            if (inputImage.type() == CV_8UC1) {
                cv::Mat imageIntegral(inputImage.cols + 1,inputImage.rows + 1,CV_32FC1);
                cv::integral(inputImage, imageIntegral);
                inputIntegralImage = imageIntegral.clone();
                imageIntegral.release();
            } else {
                printf("Error getHashImage: The input video format inappropriate.");
                return 0;
            }
        } else {
            inputIntegralImage = inputImage.clone();
        }

        short x2,y3;
        //int const imageWidth(image.cols), imageHeight(image.rows);
        // точки пр€моугольника
        x2 = rect.x + rect.width - 1;
        y3 = rect.y + rect.height - 1;

        unsigned long *const lineH( inputIntegralImage.ptr<unsigned long>(rect.y - 1) );
        unsigned long *const lineL( inputIntegralImage.ptr<unsigned long>(y3) );
        float mean = (float)lineH[rect.x - 1] + (float)lineL[x2]
                    -(float)lineH[x2] -  (float)lineL[rect.x - 1];
        float kMean = rect.width * rect.height;
        mean = mean / kMean;
        *outMean = (unsigned char)mean;

        float data; // €ркость одного пиксел€ сетки wHash * hHash
        short incX = rect.width / wHash; // инкремент по оси X
        short incY = rect.height / hHash;
        if (incX == 0 || incY == 0 ) {
            printf("Error getHashImage: width or height of the bounding box is too small.");
            return 0;
        }
        float kMeanShRect = incX * incY; // количесвто точек на пиксель сетки
        short rectX2, rectY3; // точки ограничительной рамки
        unsigned long hashImage = 0; // хэш

        int i = 0; // позици€ бита хэша

        int const maxY = rect.y + incY * hHash - 1; // максимальное значение Y на изображении
        int const maxX = rect.x + incX * wHash - 1; // максимальное значение X на изображении

        for (short y = rect.y - 1; y < maxY; y += incY) {
            rectY3 = y + incY;
            unsigned long *const lineH( inputIntegralImage.ptr<unsigned long>(y) );
            unsigned long *const lineL( inputIntegralImage.ptr<unsigned long>(rectY3) );
            for (short x = rect.x - 1; x < maxX; x += incX) {
                rectX2 = x + incX;
                data = (float)lineH[x] + (float)lineL[rectX2]
                     - (float)lineH[rectX2] - (float)lineL[x];
                data = data / kMeanShRect;
                hashImage |= (data > mean) ? tableHashBit[i] : 0;
                i++;
            }
        }
        inputIntegralImage.release();
        return hashImage;
    }

    // рекурсивна€ функци€ дл€ создани€ шума в пределах рассто€ни€ ’эмминга.
    static void _getNoise32(unsigned long hashImage, unsigned char* hashArray, unsigned char hashData, unsigned char start, unsigned char k, unsigned char maxBit) {
        if (!k)
            return;
        for (unsigned char i = start; i < maxBit; i++) {
            unsigned char _k = k;
            unsigned long _hash = hashImage ^ (0x0000000000000001 << i);
            _k--;
            if (_k) {
                _getNoise32(_hash, hashArray, hashData, i + 1, _k, maxBit);
            } else {
                hashArray[_hash] |= hashData;
            }
        }
    }

    void getNoise32(unsigned long hashImage, unsigned char* hashArray, unsigned char hashData, unsigned char hammingDistance, unsigned char maxBit) {
        for (unsigned char d = 1; d <= hammingDistance; d++) {
            _getNoise32(hashImage, hashArray, hashData, 0, d, maxBit);
        }
    }

    void getMeanNoise(unsigned char* meanArray, unsigned char mean, unsigned char noise) {
        short _start = mean - noise;
        short _end = mean + noise;
        if (_start < 0) {
            _start = 0;
        }
        if (_end > 0) {
            _end = 0;
        }
        for (unsigned char i = _start; i < mean; i++) {
            meanArray[i] = 1;
        }
        for (unsigned char i = mean + 1; i <= _end; i++) {
            meanArray[i] = 1;
        }
    }

    void showHash32(unsigned long hashImage, cv::Mat& image, int hashW, int hashH, int w, int h, bool isRGB = false) {
        cv::Mat grayImage(cv::Size(hashW, hashH), CV_8UC1);
        cv::Mat resizeImage(cv::Size(w, h), CV_8UC1);
        for (int y = 0; y < hashH; ++y) {
            for (int x = 0; x < hashW; ++x) {
                if ((hashImage & 0x0000000000000001) == 0x0000000000000001)
                    grayImage.at<unsigned char>(y, x) = 255;
                else
                    grayImage.at<unsigned char>(y, x) = 0;
                hashImage = hashImage >> 1;
            }
        }
        cv::resize(grayImage, resizeImage, cv::Size(w, h), 0, 0, CV_INTER_AREA);
        if (!isRGB) {
            image = resizeImage.clone();
        } else {
            cv::Mat colorImage;
            cv::cvtColor(resizeImage, colorImage, CV_GRAY2BGR ); // делаем изображение цветным
            image = colorImage.clone();
            colorImage.release();
        }
        grayImage.release();
        resizeImage.release();
    }
}
