/*
// Copyright (C) 2020-2023 Intel Corporation
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

#include "models/image_encodings_model.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"
#include "utils/slog.hpp"


std::string ImageEncodingsModel::ModelType = "ImageEncoder";

ImageEncodingsModel::ImageEncodingsModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration) : ImageModel(model, configuration) {

    if (model->has_rt_info("model_info", "mean_values")) {
        const std::string& str = model->get_rt_info<std::string>("model_info", "mean_values");
        if (str.empty()) {
            throw std::runtime_error("Image encoder requires mean values");
        } else {
            mean_values = model->get_rt_info<std::vector<float>>("model_info", "mean_values");
        }
    }
}

ImageEncodingsModel::ImageEncodingsModel(std::shared_ptr<InferenceAdapter>& adapter)
        : ImageModel(adapter) {
    const ov::AnyMap& configuration = adapter->getModelConfig();
    auto mean_values_iter = configuration.find("blur_strength");
    if (mean_values_iter != configuration.end()) {
        mean_values = mean_values_iter->second.as<std::vector<float>>();
    }
    auto scale_values_iter = configuration.find("scale_values");
    if (scale_values_iter != configuration.end()) {
        scale_values = scale_values_iter->second.as<std::vector<float>>();
    }
}

std::unique_ptr<ImageEncodingsModel> ImageEncodingsModel::create_model(const std::string& modelFile, const ov::AnyMap& configuration, bool preload, const std::string& device) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(modelFile);

    // Check model_type in the rt_info, ignore configuration
    std::string model_type = ImageEncodingsModel::ModelType;
    try {
        if (model->has_rt_info("model_info", "model_type")) {
            model_type = model->get_rt_info<std::string>("model_info", "model_type");
        }
    } catch (const std::exception&) {
        slog::warn << "Model type is not specified in the rt_info, use default model type: " << model_type << slog::endl;
    }

    if (model_type != ImageEncodingsModel::ModelType) {
        throw std::runtime_error("Incorrect or unsupported model_type is provided in the model_info section: " + model_type);
    }

    std::unique_ptr<ImageEncodingsModel> segmentor{new ImageEncodingsModel(model, configuration)};
    segmentor->prepare();
    if (preload) {
        segmentor->load(core, device);
    }
    return segmentor;
}

std::unique_ptr<ImageEncodingsModel> ImageEncodingsModel::create_model(std::shared_ptr<InferenceAdapter>& adapter) {
    const ov::AnyMap& configuration = adapter->getModelConfig();
    auto model_type_iter = configuration.find("model_type");
    std::string model_type = ImageEncodingsModel::ModelType;
    if (model_type_iter != configuration.end()) {
        model_type = model_type_iter->second.as<std::string>();
    }

    if (model_type != ImageEncodingsModel::ModelType) {
        throw std::runtime_error("Incorrect or unsupported model_type is provided: " + model_type);
    }

    std::unique_ptr<ImageEncodingsModel> segmentor{new ImageEncodingsModel(adapter)};
    return segmentor;
}

void ImageEncodingsModel::updateModelInfo() {
    ImageModel::updateModelInfo();
    model->set_rt_info(ImageEncodingsModel::ModelType, "model_info", "model_type");
}

void ImageEncodingsModel::prepareInputsOutputs(
    std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input  -------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Segmentation model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputNames.push_back(input.get_any_name());
    const ov::Layout& inputLayout = getInputLayout(input);
    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();

    model = ImageModel::embedProcessing(model,
                                    inputNames[0],
                                    inputLayout,
                                    resizeMode,
                                    interpolationMode,
                                    ov::Shape{inputShape[ov::layout::width_idx(inputLayout)],
                                                inputShape[ov::layout::height_idx(inputLayout)]},
                                    pad_value,
                                    reverse_input_channels,
                                    mean_values,
                                    scale_values);

    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.output().model().set_layout(getLayoutFromShape(model->output().get_partial_shape()));
    ppp.output().tensor().set_element_type(ov::element::f32).set_layout("NCHW");
    model = ppp.build();
    embedded_processing = true;

    outputNames.push_back(model->output().get_any_name());
}

std::unique_ptr<ResultBase> ImageEncodingsModel::postprocess(InferenceResult& infResult) {
    InferenceResult* result = new InferenceResult(infResult);
    return std::unique_ptr<ResultBase>(result);
}

std::unique_ptr<InferenceResult>
ImageEncodingsModel::infer(const ImageInputData& inputData) {
    auto result = ModelBase::infer(static_cast<const InputData&>(inputData));
    return std::unique_ptr<InferenceResult>(static_cast<InferenceResult*>(result.release()));
}
