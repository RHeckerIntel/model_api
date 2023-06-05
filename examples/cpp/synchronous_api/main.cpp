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
#include <cmath>
#include <memory>
#include <opencv2/core/cvdef.h>
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

struct InteractivePoint: public cv::Point {
    InteractivePoint(int x, int y, bool positive): cv::Point(x, y), positive(positive) {}
    bool positive;
};

std::vector<InteractivePoint> points = {};
std::vector<cv::Rect> boxes = {};
cv::Mat image;
cv::Size inputSize(1024, 1024);
ov::Tensor image_embeddings_tensor;


PointsInput buildPointCoords() {
    struct PointsInput input;
    for (auto &point: points) {
        input.point_coords.emplace_back(point.x);
        input.point_coords.emplace_back(point.y);
        input.labels.emplace_back(point.positive);
    }

    // no boxes yet...
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

    std::cout << "point coords: ";
    for (auto v: input.point_coords) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;

    std::cout << "labels: ";
    for (auto v: input.labels) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;

    return input;
}

void runMask() {
    std::string device = "AUTO";


    ov::Core core;
    auto model = core.read_model("/data/sam/sam_mask_predictor.xml");
    auto inferenceAdapter = std::make_shared<OpenVINOInferenceAdapter>();
    inferenceAdapter->loadModel(model, core, device);

    auto inputNames = inferenceAdapter->getInputNames();

    for (auto name: inputNames) {
            std::cout << name << std::endl;
            std::cout << inferenceAdapter->getInputShape(name) << std::endl;
    }

    auto point_data = buildPointCoords();
    ov::Tensor point_coords(ov::element::f32, ov::Shape{1, point_data.point_coords.size() / 2, 2}, point_data.point_coords.data());
    ov::Tensor point_labels(ov::element::f32, ov::Shape{1, point_data.labels.size()}, point_data.labels.data());

    std::cout << "running mask..." << std::endl;

    InferenceInput inputs;
    InferenceResult result;
    inputs.emplace("image_embeddings", image_embeddings_tensor);
    inputs.emplace("point_coords", point_coords);
    inputs.emplace("point_labels", point_labels);
    for (auto &input: inputs) {
        std::cout << input.first << ":" << input.second.get_shape() << std::endl;
    }
    result.outputsData = inferenceAdapter->infer(inputs);

    auto outputNames = inferenceAdapter->getOutputNames();

    auto mask_tensor = result.outputsData["masks"];
    std::cout << mask_tensor.get_shape() << std::endl;

    cv::Mat predictions(1024, 1024, CV_32FC1, mask_tensor.data<float_t>());
    cv::imshow("predictions", predictions);
    for (auto name: outputNames) {
            std::cout << name << std::endl;
            std::cout << result.outputsData[name].get_shape() << std::endl;
            //std::cout << result.getFirstOutputTensor().get_shape() << std::endl;
    }

    float* iou_predictions = result.outputsData["iou_predictions"].data<float_t>();
    std::cout << iou_predictions[0] << std::endl;

    //std::cout << result.outputsData["mask"].get_shape() << std::endl;

    //auto embedding_processing_model = embedProcessing(ov_encoder_model, "input.1", "NCHW", RESIZE_FILL, cv::INTER_LINEAR, ov::Shape{1024, 1024}, 0, true, mean, scale, typeid(ov::element::f32));
}

/*
 *moreover, we identified and selected
only stable masks (we consider a mask stable if thresholding the probability map at 0.5 − δ and 0.5 + δ results in
similar masks). F
 * */

cv::Point pointDown;
cv::Point pointUp;

bool PointMode = false;

static void onMouse( int event, int x, int y, int, void* )
{
    if (PointMode) {
        if(!(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN))
            return;

        bool positive = event == cv::EVENT_LBUTTONDOWN;
        std::cout << x <<  ":" << y << std::endl;

        points.emplace_back(x, y, positive);

        cv::Scalar color(0, positive ? 255 : 0, positive ? 0 : 255, 0);

        cv::circle(image, {x, y}, 3, color);
        cv::imshow("image", image);
        runMask();
    } else {
        if(!(event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_LBUTTONUP))
            return;

        if (event == cv::EVENT_LBUTTONDOWN) {
            pointDown = cv::Point(x, y);
        } else {
            pointUp = cv::Point(x, y);

            cv::Rect box(pointDown, pointUp);
            boxes.push_back(box);
            cv::rectangle(image, box, 255);
            cv::imshow("image", image);
            runMask();
        }


    }
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

        std::cout << points.size() << std::endl;

        cv::Vec2f rescale(inputSize.width / (float)image.cols, inputSize.height / (float)image.rows);

        cv::resize(image, image, inputSize);


        auto model = ImageEncodingsModel::create_model(argv[1]);
        auto result = model->infer(image);
        image_embeddings_tensor = result->getFirstOutputTensor();

        cv::namedWindow( "image", 0);
        cv::imshow( "image", image);
        cv::setMouseCallback( "image", onMouse, 0 );

        //runMask();

        cv:: waitKey( 0 );



        //encoderPOC(argv[1], image);

        //std::cout << image_embeddings_tensor.get_element_type() << std::endl;
        //int shape[4] = { 1, 256, 64, 64};
        //cv::Mat image_embeddings_mat(4, shape, CV_32F, image_embeddings_tensor.data<float>());

        //std::cout << image_embeddings_mat.size << std::endl;

        //if (outChannels == 1 && outTensor.get_element_type() == ov::element::i32) {
        //cv::Mat image_embeddings_mat(4, CV_32SC1, outTensor.data<int32_t>());

        //cv::FileStorage file("/data/sam/cattle_encodings.ext", cv::FileStorage::WRITE);
        //file << "cattle_encodings" << image_embeddings_mat;

        //file.release();

        //embedding_processing_model
        // -----------------------------------
        //cv::imwrite("/data/preprocessed.png", transform(image));

    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    return 0;
}
