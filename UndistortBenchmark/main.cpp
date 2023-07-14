#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>

int main()
{
    // Define the checkerboard size
    int width = 320;
    int height = 240;

    double fx, fy, cx, cy;
    double k1, k2, k3, k4, k5, k6;
    double p1, p2;
    double s1, s2, s3, s4;
    
    fx = 320;
    fy = 240;
    cx = 160;
    cy = 120;
    k1 = 0.8;
    k2 = 0.0;
    k3 = 0.0;
    k4 = 0.0;
    k5 = 0.0;
    k6 = 0.0;
    p1 = 0.0;
    p2 = 0.0;
    s1 = 0.0;
    s2 = 0.0;
    s3 = 0.0;
    s4 = 0.0;

    // Create a blank image
    cv::Mat checkerboard(height, width, CV_8UC1);

    // Generate the checkerboard pattern
    int squareSize = 40;  // Size of each square in the checkerboard
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int row = i / squareSize;
            int col = j / squareSize;
            if ((row + col) % 2 == 0)
                checkerboard.at<uchar>(i, j) = 255;  // White color
            else
                checkerboard.at<uchar>(i, j) = 0;    // Black color
        }
    }

    // Load the camera calibration parameters (intrinsic and distortion coefficients)
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        fx, 0,  cx,
        0,  fy, cy,
        0,  0,  1);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 12) <<
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4);
    
    cv::Mat remap_map1(height, width, CV_32FC1);  // Initialize remap_map1
    cv::Mat remap_map2(height, width, CV_32FC1);  // Initialize remap_map2
    
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    
    std::cout << R << std::endl;

    cv::Mat outputImage(height, width, CV_8UC1);  // Allocate memory for the output image

    // Perform multiple undistortions and measure the average execution time
    int numIterations = 0;  // Number of undistortions that have been performed
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end = start + std::chrono::seconds(100);
    while (std::chrono::high_resolution_clock::now() < end) {
        // Calculate the remap maps
        cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, cv::Size(width, height), CV_32FC2, remap_map1, remap_map2);
        // Split the undistortion process across multiple CPU cores
        int numThreads = std::thread::hardware_concurrency();
        std::vector<std::future<void>> futures;
        int rowsPerThread = height / numThreads;

        for (int t = 0; t < numThreads; t++)
        {
            int startRow = t * rowsPerThread;
            int endRow = startRow + rowsPerThread;

            futures.push_back(std::async(std::launch::async, [startRow, endRow, rowsPerThread, &checkerboard, &outputImage, &remap_map1, &remap_map2]() {
                cv::remap(checkerboard, outputImage(cv::Range(startRow, endRow), cv::Range::all()), remap_map1(cv::Range(startRow, endRow), cv::Range::all()), remap_map2, cv::INTER_LINEAR);
            }));
        }

        for (auto& future : futures) {
            future.wait();
        }
        numIterations++;
    }
    // May have entered loop right before timeout so we take the time now to calculate complete duration
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;

    // Calculate the average execution time and frames per second (FPS)
    double averageExecutionTime = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() / numIterations;
    double averageFPS = 1.0 / averageExecutionTime;

    // Print the average execution time and FPS
    std::cout << "Average Execution Time: " << averageExecutionTime << " seconds" << std::endl;
    std::cout << "Average FPS: " << averageFPS << std::endl;
    
    // Display the original and undistorted images
    cv::imshow("Original Image", checkerboard);
    cv::imshow("Undistorted Image", outputImage);
    cv::waitKey(0);

    return 0;
}
