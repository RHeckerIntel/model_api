/*
// Copyright (C) 2018-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "adapters/openvino_adapter.h"
#include "utils/ocv_common.hpp"
#include <chrono>
#include <cmath>
#include <memory>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/persistence.hpp>
#include <openvino/runtime/core.hpp>
#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/image_encodings_model.h>
#include <models/input_data.h>
#include <models/results.h>
#include <adapters/inference_adapter.h>

struct PointsInput {
    std::vector<float> point_coords;
    std::vector<float> labels;
};

using Contours = std::vector<std::vector<cv::Point>>;

struct InteractivePoint: public cv::Point {
    InteractivePoint(int x, int y, bool positive): cv::Point(x, y), positive(positive) {}
    bool positive;
};

std::vector<InteractivePoint> points = {};
std::vector<cv::Rect> boxes = {};
cv::Mat image;
cv::Mat outputImage;
cv::Size inputSize(1024, 1024);
ov::Tensor image_embeddings_tensor;

void drawContours(Contours contours) {
    outputImage = cv::Mat(1024, 1024, CV_32FC1);
    image.copyTo(outputImage);


    for (auto point: points) {
        cv::circle(outputImage, point, 2, 255, -1);
    }

    for (auto rect: boxes) {
        cv::rectangle(outputImage, rect, 255, -1);
    }

    cv::drawContours(outputImage, contours, -1, 255);
    cv::imwrite("/data/sam_output.png", outputImage);
    cv::imshow("image", outputImage);
}

PointsInput buildPointCoords(std::vector<InteractivePoint> &points, std::vector<cv::Rect> &boxes, bool batch = false) {
    struct PointsInput input;
    for (auto &point: points) {
        input.point_coords.emplace_back(point.x);
        input.point_coords.emplace_back(point.y);
        input.labels.emplace_back(point.positive);
    }

    if (batch) {
        return input;
    }

    if (boxes.size() == 0) {
        input.point_coords.emplace_back(0);
        input.point_coords.emplace_back(0);
        input.labels.emplace_back(-1);
    } else {
        for (auto &box: boxes) {
            input.point_coords.emplace_back(box.tl().x);
            input.point_coords.emplace_back(box.tl().y);
            input.labels.emplace_back(2);
            input.point_coords.emplace_back(box.br().x);
            input.point_coords.emplace_back(box.br().y);
            input.labels.emplace_back(3);
        }
    }
    return input;
}

template <typename T> std::vector<std::vector<T>> batchVector(std::vector<T> objects, int batchSize) {
    std::vector<std::vector<T>> batches;
    for (size_t i = 0; i < objects.size() / batchSize; i++ ) {
        std::vector<T> batch;
        auto start = objects.begin() + i * batchSize;
        int endIndex = std::min<int>((i + 1) * batchSize, objects.size());
        auto end = objects.begin() + endIndex;
        batch.insert(batch.begin(), start, end);
        batches.push_back(batch);
    }
    return batches;
}

float calculateStabilityScore(cv::Mat mask) {
    float mask_threshold = 0;
    float stability_score_offset = 1.0f;


    cv::Mat unionMask;
    cv::threshold(mask, unionMask, mask_threshold - stability_score_offset, 100.0f, cv::THRESH_BINARY);
    cv::Mat intersectionMask;
    cv::threshold(mask, intersectionMask, mask_threshold + stability_score_offset, 100.0f, cv::THRESH_BINARY);

    return cv::countNonZero(intersectionMask) / (float)cv::countNonZero(unionMask);
}

class MaskPredictor {
public:
    std::string device = "AUTO";
    ov::Core core;
    float stability_threshold = 0.9f;
    std::shared_ptr<InferenceAdapter> inferenceAdapter = std::make_shared<OpenVINOInferenceAdapter>();

    MaskPredictor() {}
    MaskPredictor(std::string model_file) {
        auto model = core.read_model(model_file);
        inferenceAdapter->loadModel(model, core, device);
    }

    Contours runMask(std::vector<InteractivePoint> &points, std::vector<cv::Rect> &boxes) {
        auto point_data = buildPointCoords(points, boxes);
        ov::Tensor point_coords(ov::element::f32, ov::Shape{1, point_data.point_coords.size() / 2, 2}, point_data.point_coords.data());
        ov::Tensor point_labels(ov::element::f32, ov::Shape{1, point_data.labels.size()}, point_data.labels.data());
        auto mask_tensor = infer(point_coords, point_labels);


        int batchSize = mask_tensor.get_shape()[0];
        int stride = mask_tensor.get_strides()[0];

        std::vector<cv::Mat> masks = {};
        for (int i = 0; i < batchSize; i++) { // ???? no batch no need
            cv::Mat new_mask(1024, 1024, CV_32FC1, mask_tensor.data<float_t>() + (i * stride / sizeof(float_t)));
            //cv::threshold(new_mask, new_mask, 0.5, 1, cv::THRESH_BINARY);
            if (calculateStabilityScore(new_mask) > stability_threshold) {
                new_mask.convertTo(new_mask, CV_8UC1, 255);
                combineNewMask(masks, new_mask);
            }
        }

        if (masks.empty()) {
            return {};
        }

        return getContours(masks);
    }

    Contours batchRunMask(std::vector<InteractivePoint> &points, std::vector<cv::Rect> boxes, int batchSize) {
        std::vector<cv::Mat> masks = {};

        auto batches = batchVector<InteractivePoint>(points, batchSize);
        for (size_t i = 0; i < batches.size(); i++ ) {
            std::cout << "Batch " << i << "/" << batches.size() << std::endl;
            auto point_data = buildPointCoords(batches[i], boxes, true);
            ov::Tensor point_coords(ov::element::f32, ov::Shape{point_data.point_coords.size() / 2, 1, 2}, point_data.point_coords.data());
            ov::Tensor point_labels(ov::element::f32, ov::Shape{point_data.labels.size(), 1}, point_data.labels.data());
            auto mask_tensor = infer(point_coords, point_labels);

            int batchSize = mask_tensor.get_shape()[0];
            int stride = mask_tensor.get_strides()[0];

            for (int i = 0; i < batchSize; i++) {
                cv::Mat new_mask(1024, 1024, CV_32FC1, mask_tensor.data<float_t>() + (i * stride / sizeof(float_t)));
                if (calculateStabilityScore(new_mask) > stability_threshold) {
                    new_mask.convertTo(new_mask, CV_8UC1, 255);
                    combineNewMask(masks, new_mask);
                }
            }
        }

        std::cout << "masks found: " << masks.size() << std::endl;
        return getContours(masks);
    }

    ov::Tensor infer(ov::Tensor point_coords, ov::Tensor point_labels) {
        InferenceInput inputs;
        InferenceResult result;
        inputs.emplace("image_embeddings", image_embeddings_tensor);
        inputs.emplace("point_coords", point_coords);
        inputs.emplace("point_labels", point_labels);
        auto start = std::chrono::high_resolution_clock::now();
        result.outputsData = inferenceAdapter->infer(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "time: " <<  std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << std::endl;

        float* iou_predictions = result.outputsData["iou_predictions"].data<float_t>();
        std::cout << iou_predictions[0] << std::endl;

        return result.outputsData["masks"];
    }

    void combineNewMask(std::vector<cv::Mat> &saved_masks, cv::Mat mask) {
        for (auto &saved_mask: saved_masks) {
            cv::Mat intersectionMat(inputSize, CV_8UC1);
            cv::Mat unionMat(inputSize, CV_8UC1);
            intersectionMat = mask.mul(saved_mask);
            unionMat = saved_mask + mask;
            float iou = cv::countNonZero(intersectionMat) / (float)cv::countNonZero(unionMat);
            if (iou > 0.5f) {
                unionMat.copyTo(saved_mask);
                return;
            }
        }
        std::cout << "storing a mask..." << std::endl;
        saved_masks.push_back(mask);
    }

    std::vector<std::vector<cv::Point>> getContours(std::vector<cv::Mat> masks) {
        Contours contours;
        for (auto &mask: masks) {
            Contours newContours;
            cv::findContours(mask, newContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            contours.insert(contours.end(), newContours.begin(), newContours.end());
        }

        return contours;
    }
};

MaskPredictor maskPredictor;


/*
 *moreover, we identified and selected
only stable masks (we consider a mask stable if thresholding the probability map at 0.5 − δ and 0.5 + δ results in
similar masks). F
 * */

cv::Point pointDown;
cv::Point pointUp;

bool PointMode = true;

static void onMouse( int event, int x, int y, int, void* )
{
    if (PointMode) {
        if(!(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN))
            return;

        points = {};
        bool positive = event == cv::EVENT_LBUTTONDOWN;
        std::cout << x <<  ":" << y << std::endl;
        points.emplace_back(x, y, positive);

        drawContours(maskPredictor.runMask(points,boxes));

    } else {
        if(!(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_LBUTTONUP))
            return;

        if (event == cv::EVENT_LBUTTONDOWN) {
            pointDown = cv::Point(x, y);
        } else {
            pointUp = cv::Point(x, y);

            points = {};
            boxes = {};
            boxes.emplace_back(pointDown, pointUp);
            drawContours(maskPredictor.runMask(points,boxes));


        }
    }
}

void segment_all() {
    float n = 40;
    cv::Point template_size(inputSize.width / n, inputSize.height / n);

    cv::Point grid_point;
    image.copyTo(outputImage);
    points = {};
    boxes = {};
    if (PointMode) {
        for (grid_point.y = template_size.y / 2; grid_point.y < inputSize.height; grid_point.y += template_size.y){
            for (grid_point.x = template_size.x / 2; grid_point.x < inputSize.width; grid_point.x += template_size.x) {
                points.emplace_back(grid_point.x, grid_point.y, true);
            }
        }
    } else {
        for (grid_point.y = 0; grid_point.y < inputSize.height; grid_point.y += template_size.y){
            for (grid_point.x = 0; grid_point.x < inputSize.width; grid_point.x += template_size.x) {
                boxes.emplace_back(grid_point, grid_point + template_size);
            }
        }
    }
    auto contours = maskPredictor.batchRunMask(points, boxes, 20);
    drawContours(contours);
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage : " << argv[0] << " <path_to_model> <path_to_image>"
                      << std::endl;
            return EXIT_FAILURE;
        }

        image = cv::imread(argv[2]);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }

        cv::Vec2f rescale(inputSize.width / (float)image.cols, inputSize.height / (float)image.rows);
        outputImage = cv::Mat(1024, 1024, CV_32FC1);

        cv::resize(image, image, inputSize);

        auto model = ImageEncodingsModel::create_model(argv[1]);
        auto result = model->infer(image);
        image_embeddings_tensor = result->getFirstOutputTensor();

        cv::namedWindow( "image", 0);
        cv::imshow( "image", image);
        cv::setMouseCallback( "image", onMouse, 0 );

        maskPredictor = MaskPredictor("/data/sam/sam_mask_predictor.xml");
        segment_all();
        cv:: waitKey( 0 );

    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    return 0;
}
