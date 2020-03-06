/**
  \ingroup TensorflowCppWrapper
  \file    tf_image.hpp
  \brief   This tf_image.hpp file contains the function signatures for workign with image prediction in Tensorflow
  \author  kovalenko
  \date    2020-03-05

  Copyright:
  2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
  The copyright of this software source code is the property of HHI.
  This software may be used and/or copied only with the written permission
  of HHI and in accordance with the terms and conditions stipulated
  in the agreement/contract under which the software has been supplied.
  The software distributed under this license is distributed on an "AS IS" basis,

  This file shall not be used for any implementation stuff.
  This is the file included by users of the lib PythonWrapper
 */

#pragma once

#include "tf_utils.hpp"

#include <map>
#include <unordered_map>

#include <opencv2/opencv.hpp>

namespace tf_image { 
  
  class TF_Model {

  private:
    TF_Session* m_pSession;
    TF_Graph* m_pGraph;
    TF_Status* m_pStatus;

    std::vector<TF_Output> input_ops;
    std::vector<TF_Output> output_ops;

    std::vector<TF_Tensor*> processDataImg( const std::map<std::string, std::vector<cv::Mat>>& input );

    std::unordered_map<std::string, TF_Output> input_op_names;
    std::unordered_map<std::string, TF_Output> output_op_names;

    std::vector<int64_t> GetTensorShape( TF_Tensor* tensor );

  public:
    TF_Model() = default;

    ~TF_Model();

    /**
      Load a model, creating the graph and the session
      \param[in] path Path to the model file
      \param[in] gpu_memory_fraction Fraction of the GPU memory that will be allocated for the model
      \param[in] inferIO Allow the model to automatically try to infer the names of the Input and Output operations/layers
    */
    bool loadModel( const std::string& path, const double gpu_memory_fraction = 1.0f, bool inferIO = false );

    void setInputs( const std::vector<std::string>& inputNames );
    void setOutputs( const std::vector<std::string>& outputNames );   

    /**
      Run prediction on an image
      \param[in] input A vector of OpenCV images: one image per network's every input layer/operation.
      \retval A vector of prediction results: one per the network's evey output layer/operation
    */
    template<class T>
    std::vector<T> predict( const std::vector<cv::Mat>& input );

  private:     

    /**
      Process images, expecting a float vector as output
      \param[in] input Input data as a vector of images [ image_0, image_1, ..., image_# ], one image per network's input
      \retval Prediction results as a vector of vectors, one per network output [ vector_0, vector_1, ..., vector_# ]
    */
    std::vector< std::vector<float> > predict_image2vector( const std::vector<cv::Mat>& input );

    /**
      Process images, expecting also images as output
      \param[in] input Input data as a vector of images [ image_0, image_1, ..., image_# ], one image per network's input
      \retval Prediction results as a vector of images, one per network output [ image_0, image_1, ..., image_# ]
    */
    std::vector< cv::Mat > predict_image2image( const std::vector<cv::Mat>& input );

   

  };
}

/**
  Process images, expecting a float vector as output
  \param[in] input Input data as a vector of images [ image_0, image_1, ..., image_# ], one image per network's input
  \retval Prediction results as a vector of vectors, one per network output [ vector_0, vector_1, ..., vector_# ]
*/
template<> inline std::vector< std::vector<float> > tf_image::TF_Model::predict( const std::vector<cv::Mat>& input ) {
  return predict_image2vector( input );
}

/**
  Process images, expecting also images as output
  \param[in] input Input data as a vector of images [ image_0, image_1, ..., image_# ], one image per network's input
  \retval Prediction results as a vector of images, one per network output [ image_0, image_1, ..., image_# ]
*/
template<> inline std::vector< cv::Mat > tf_image::TF_Model::predict( const std::vector<cv::Mat>& input ) {
  return predict_image2image( input );
}
