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


std::shared_ptr<ov::Model> embedProcessing(std::shared_ptr<ov::Model>& model,
                                            const std::string& inputName,
                                            const ov::Layout& layout,
                                            const RESIZE_MODE resize_mode,
                                            const cv::InterpolationFlags interpolationMode,
                                            const ov::Shape& targetShape,
                                            uint8_t pad_value,
                                            bool brg2rgb,
                                            const std::vector<float>& mean,
                                            const std::vector<float>& scale,
                                            const std::type_info& dtype) {
    ov::preprocess::PrePostProcessor ppp(model);

    // Change the input type to the 8-bit image
    if (dtype == typeid(int)) {
        ppp.input(inputName).tensor().set_element_type(ov::element::u8);
    }

    ppp.input(inputName).tensor().set_layout(ov::Layout("NHWC")).set_color_format(
        ov::preprocess::ColorFormat::BGR
    );

    if (resize_mode != NO_RESIZE) {
        ppp.input(inputName).tensor().set_spatial_dynamic_shape();
        // Doing resize in u8 is more efficient than FP32 but can lead to slightly different results
        ppp.input(inputName).preprocess().custom(
            createResizeGraph(resize_mode, targetShape, interpolationMode, pad_value));
    }

    ppp.input(inputName).model().set_layout(ov::Layout(layout));

    // Handle color format
    if (brg2rgb) {
        ppp.input(inputName).preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
    }

    ppp.input(inputName).preprocess().convert_element_type(ov::element::f32);

    if (!mean.empty()) {
        ppp.input(inputName).preprocess().mean(mean);
    }
    if (!scale.empty()) {
        ppp.input(inputName).preprocess().scale(scale);
    }

    return ppp.build();
}


void encoderPOC(std::string modelPath, cv::Mat image){
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    std::string device = "AUTO";


    ov::Core core;
    auto ov_encoder_model = core.read_model(modelPath);
    //auto model = compiled_model.get_runtime_model();
    std::vector<float> mean = {58.395, 57.12, 57.375};
    std::vector<float> scale = { 123.675, 116.28, 103.53};
    // ----------------------------------- attempt using prepostprocessor
    auto embedding_processing_model = embedProcessing(ov_encoder_model, "input.1", "NCHW", RESIZE_FILL, cv::INTER_LINEAR, ov::Shape{1024, 1024}, 0, true, mean, scale, typeid(ov::element::f32));
    //std::shared_ptr<ov::Model>
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(embedding_processing_model);
    ppp.output().model().set_layout(getLayoutFromShape(ov_encoder_model->output().get_partial_shape()));
    ppp.output().tensor().set_element_type(ov::element::f32).set_layout("NCHW");
    embedding_processing_model = ppp.build();

    auto inferenceAdapter = std::make_shared<OpenVINOInferenceAdapter>();
    inferenceAdapter->loadModel(embedding_processing_model, core, device);

    image.convertTo(image, CV_32F);
    InputTransform transform(true, "123.675 116.28 103.53","58.395 57.12 57.375");

    cv::resize(image, image, cv::Size(1024, 1024));

    InferenceInput inputs;
    InferenceResult result;
    inputs.emplace("input.1",wrapMat2Tensor(image));
    result.outputsData = inferenceAdapter->infer(inputs);
    std::cout << result.getFirstOutputTensor().get_shape() << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage : " << argv[0] << " <path_to_model> <path_to_image>"
                      << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat image = cv::imread(argv[2]);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }

        //encoderPOC(argv[1], image);

        auto model = ImageEncodingsModel::create_model(argv[1]);
        auto result = model->infer(image);
        std::cout << result->getFirstOutputTensor().get_shape() << std::endl;


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
