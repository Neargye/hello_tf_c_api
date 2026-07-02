// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2026 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include <scope_guard.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

TF_Operation* FinishOperation(TF_OperationDescription* desc, TF_Status* status) {
  auto op = TF_FinishOperation(desc, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Failed to finish operation: " << TF_Message(status) << std::endl;
    return nullptr;
  }

  return op;
}

TF_Operation* AddPlaceholder(TF_Graph* graph, const char* name, TF_DataType data_type, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", data_type);

  return FinishOperation(desc, status);
}

TF_Operation* AddIdentity(TF_Graph* graph, const char* name, TF_Output input, TF_DataType data_type, TF_Status* status) {
  auto desc = TF_NewOperation(graph, "Identity", name);
  TF_AddInput(desc, input);
  TF_SetAttrType(desc, "T", data_type);

  return FinishOperation(desc, status);
}

bool AlmostEqual(float lhs, float rhs) {
  return std::fabs(lhs - rhs) < 1.0e-6f;
}

bool WriteSampleImage(const std::string& path) {
  cv::Mat image(2, 2, CV_8UC3);
  image.at<cv::Vec3b>(0, 0) = cv::Vec3b{0, 127, 255};   // BGR.
  image.at<cv::Vec3b>(0, 1) = cv::Vec3b{64, 128, 192};
  image.at<cv::Vec3b>(1, 0) = cv::Vec3b{255, 0, 32};
  image.at<cv::Vec3b>(1, 1) = cv::Vec3b{16, 240, 80};

  return cv::imwrite(path, image);
}

std::vector<float> FlattenFloatImage(const cv::Mat& image) {
  if (!image.isContinuous()) {
    return FlattenFloatImage(image.clone());
  }

  const auto* begin = image.ptr<float>();
  const auto* end = begin + image.total() * static_cast<std::size_t>(image.channels());

  return {begin, end};
}

} // namespace

int main(int argc, char** argv) {
  const std::string image_path = argc > 1 ? argv[1] : "opencv_image_file_example.png";
  if (argc <= 1 && !WriteSampleImage(image_path)) {
    std::cout << "Failed to write sample image file" << std::endl;
    return 1;
  }

  auto bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr_image.empty()) {
    std::cout << "Failed to read image file: " << image_path << std::endl;
    return 2;
  }

  cv::Mat rgb_image;
  cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat float_image;
  rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

  const std::vector<std::int64_t> image_dims = {
    1,
    static_cast<std::int64_t>(float_image.rows),
    static_cast<std::int64_t>(float_image.cols),
    static_cast<std::int64_t>(float_image.channels()),
  };
  const auto image_values = FlattenFloatImage(float_image);

  auto status = TF_NewStatus();
  SCOPE_EXIT{ TF_DeleteStatus(status); };

  auto graph = TF_NewGraph();
  SCOPE_EXIT{ tf_utils::DeleteGraph(graph); };

  auto input = AddPlaceholder(graph, "input_image", TF_FLOAT, status);
  if (input == nullptr) {
    return 3;
  }

  auto output = AddIdentity(graph, "output_image", TF_Output{input, 0}, TF_FLOAT, status);
  if (output == nullptr) {
    return 4;
  }

  auto input_tensor = tf_utils::CreateTensor(TF_FLOAT, image_dims, image_values);
  SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); };
  if (input_tensor == nullptr) {
    std::cout << "Failed to create image tensor" << std::endl;
    return 5;
  }

  auto session = tf_utils::CreateSession(graph, status);
  SCOPE_EXIT{ tf_utils::DeleteSession(session); };
  if (session == nullptr || TF_GetCode(status) != TF_OK) {
    std::cout << "Failed to create session: " << TF_Message(status) << std::endl;
    return 6;
  }

  const std::vector<TF_Output> inputs = {TF_Output{input, 0}};
  const std::vector<TF_Tensor*> input_tensors = {input_tensor};
  const std::vector<TF_Output> outputs = {TF_Output{output, 0}};
  std::vector<TF_Tensor*> output_tensors = {nullptr};
  SCOPE_EXIT{ tf_utils::DeleteTensors(output_tensors); };

  const auto code = tf_utils::RunSession(session, inputs, input_tensors, outputs, output_tensors, status);
  if (code != TF_OK) {
    std::cout << "Failed to run session: " << TF_Message(status) << std::endl;
    return 7;
  }

  const auto result = tf_utils::GetTensorData<float>(output_tensors[0]);
  if (result.size() != image_values.size()) {
    std::cout << "Unexpected output image size" << std::endl;
    return 8;
  }

  for (std::size_t i = 0; i < image_values.size(); ++i) {
    if (!AlmostEqual(result[i], image_values[i])) {
      std::cout << "Unexpected output image value for element: " << i << std::endl;
      return 9;
    }
  }

  std::cout << "Read image file: " << image_path << std::endl;
  std::cout << "Image tensor NHWC: "
            << image_dims[0] << "x" << image_dims[1] << "x" << image_dims[2] << "x" << image_dims[3] << std::endl;
  std::cout << "First pixel normalized RGB: "
            << result[0] << ", " << result[1] << ", " << result[2] << std::endl;
  std::cout << "Processed OpenCV image file successfully" << std::endl;

  return 0;
}
