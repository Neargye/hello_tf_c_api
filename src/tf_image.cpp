/**
  \ingroup TensorflowCppWrapper
  \file    tf_image.cpp
  \brief   This tf_image.cpp file contains the implementations of the image processing functions for Tensorflow
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

#include "tf_image.hpp"

bool tf_image::TF_Model::loadModel( const std::string & path, const double gpu_memory_fraction, bool inferIO )
{
  // create an empty status;
  m_pStatus = TF_NewStatus();

  m_pGraph = tf_utils::LoadGraph( path.c_str() );

  if ( m_pGraph == nullptr ) {
    std::cerr << "Can't load graph" << std::endl;
    return false;
  }

  m_pSession  = tf_utils::CreateSession( m_pGraph, tf_utils::CreateSessionOptions( gpu_memory_fraction ) );
  if ( m_pSession == nullptr ) {
    std::cerr << "Can't create session" << std::endl;
    return false;
  }

  if ( inferIO ) {
    // load the list of layers, try to automatically set the input and output operations
    TF_Operation* op;
    std::size_t pos = 0;

    // get first:
    op = TF_GraphNextOperation( m_pGraph, &pos );
    auto name = TF_OperationName( op );

    setInputs( { name } );

    // get last:
    while ( (op = TF_GraphNextOperation( m_pGraph, &pos )) != nullptr ) {
      name = TF_OperationName( op );
    }
    setOutputs( { name } );
  }

  return true;
}

void tf_image::TF_Model::setInputs( const std::vector<std::string>& inputNames )
{
  input_ops.clear();
  input_op_names.clear();

  for ( auto& name : inputNames ) {

    TF_Output op = { TF_GraphOperationByName( m_pGraph, name.c_str() ), 0 };
    input_ops.push_back( op );
    input_op_names[name] = op;
  }
}

void tf_image::TF_Model::setOutputs( const std::vector<std::string>& outputNames )
{
  output_ops.clear();
  output_op_names.clear();

  for ( auto& name : outputNames ) 
  {
    TF_Output op = { TF_GraphOperationByName( m_pGraph, name.c_str() ), 0 };
    output_ops.push_back( op );
    output_op_names[name] = op;
  }
}

std::vector<std::vector<float>> tf_image::TF_Model::predict_image2vector( const std::vector<cv::Mat>& input )
{
  std::vector<std::vector<float>> output;

  if ( input.size() != input_ops.size() ) {
    return output;
  }

  std::map<std::string, std::vector<cv::Mat>> inputMap;

  int j = 0;
  for ( auto& inp : input_op_names ) {
    inputMap[inp.first] = { input[j++] };
  }

  auto output_tensors = processDataImg( inputMap );

  // if everything ok, parse the output:
  if ( !output_tensors.empty() ) {

    // construct the method output:
    for ( size_t i = 0; i < output_tensors.size(); i++ )
    {
      auto shape = GetTensorShape( output_tensors.at( i ) );
      auto data = tf_utils::GetTensorsData<float>( { output_tensors.at( i ) } );
      output.push_back(data[0]);
    }
  } 

  return output;
}

std::vector<cv::Mat> tf_image::TF_Model::predict_image2image( const std::vector<cv::Mat>& input )
{
  std::vector<cv::Mat> output;

  if ( input.size() != input_ops.size() ) {
    return output;
  }

  std::map<std::string, std::vector<cv::Mat>> inputMap;

  int j = 0;
  for ( auto& inp : input_op_names ) {
    inputMap[inp.first] = { input[j++] };
  }

  auto output_tensors = processDataImg( inputMap );

  // if everything ok, parse the output:
  if ( !output_tensors.empty() ) {

    // construct the method output:
    for ( size_t i = 0; i < output_tensors.size(); i++ )
    {
      auto shape = GetTensorShape( output_tensors.at( i ) );
      auto data = tf_utils::GetTensorsData<float>( { output_tensors.at( i ) } );

      // go through all outputs           
      output.push_back( cv::Mat( static_cast<int>(shape[1]), static_cast<int>(shape[2]), CV_32FC( static_cast<int>(shape[3]) ), (void*) data[i].data() ) );      
    }
  }

  return output;
}

tf_image::TF_Model::~TF_Model()
{
  tf_utils::DeleteGraph( m_pGraph );
  TF_DeleteStatus( m_pStatus );
  tf_utils::DeleteSession( m_pSession );
}

std::vector<TF_Tensor*> tf_image::TF_Model::processDataImg( const std::map<std::string, std::vector<cv::Mat>>& input ) {

  // list of input and output tensors
  std::vector<TF_Tensor*> input_tensors;
  std::vector<TF_Tensor*> output_tensors;

  // check that the number of keys in the input map is the same as the number of input layers
  if ( input.size() != input_ops.size() ) {
    return output_tensors;
  }

  // set batch size:
  int batchSize = static_cast<int>(input.begin()->second.size());

  // list of input layers
  std::vector< std::string > inputOperationNames;

  // iterate over the inputs:
  for ( auto &inp : input_op_names ) {

    // layer name
    const std::string& layer_name = inp.first;

    // find that layer name in the function input:
    if ( input.find( layer_name ) == input.end() ) {
      // wasn't found
      return output_tensors;
    }

    // else, if found, get input batch:
    const std::vector<cv::Mat>& batch = input.at( layer_name );

    // set the input tensor's dimensions
    const std::vector<std::int64_t> input_dims = { batchSize, batch.front().rows, batch.front().cols, batch.front().channels() };

    // fill a vector with the input data
    std::vector<float> input_data;

    // convert all images to float32:
    for ( auto& image : batch ) {
      // convert to 32f
      cv::Mat image32f;
      if ( image.type() == CV_32F )
        image.copyTo( image32f );
      else
        image.convertTo( image32f, CV_32F );

      // and put into the input data vector
      input_data.insert( input_data.end(), (float*) image32f.data, (float*) image32f.data + image32f.total() * image32f.channels() );
    }

    // create a tensor and put it into the list of input tensors
    input_tensors.push_back( tf_utils::CreateTensor( TF_FLOAT, input_dims, input_data ) );
  }

  // allocate output tensors
  for ( size_t i = 0; i < output_op_names.size(); i++ )
  {
    output_tensors.push_back( nullptr );
  }

  // run inference on the network
  const TF_Code code = tf_utils::RunSession( m_pSession, input_ops, input_tensors, output_ops, output_tensors );

  if ( code == TF_OK ) { 
    return output_tensors;
  }
  else { 
    return std::vector<TF_Tensor*>();
  }
}

std::vector<int64_t> tf_image::TF_Model::GetTensorShape( TF_Tensor * tensor )
{
  auto ndims = TF_NumDims( tensor );

  std::vector<int64_t> dims;

  for ( int i = 0; i < ndims; i++ ) {
    dims.push_back( TF_Dim( tensor, i ) );
  }

  return dims;
}

