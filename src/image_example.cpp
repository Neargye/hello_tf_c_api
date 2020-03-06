/**
  \ingroup TensorflowCppWrapper
  \file    image_example.cpp
  \brief   This image_example.cpp file contains the example of using the Tensorflow API for image prediction
  \author  kovalenko
  \date    2020-03-05

  Copyright:
  2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
  The copyright of this software source code is the property of HHI.
  This software may be used and/or copied only with the written permission
  of HHI and in accordance with the terms and conditions stipulated
  in the agreement/contract under which the software has been supplied.
  The software distributed under this license is distributed on an "AS IS" basis,
  WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
*/

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "tf_image.hpp"

void run_single_image_example();
void run_expecting_image_example();
cv::Mat createHeatmap( const cv::Mat& heatmaps );

int main( ) {  

  std::cout << "Running the INPUT_IMAGE -> GET_VECTOR example:" << std::endl;
  run_single_image_example();
  std::cin.get();

  std::cout << std::endl;

  std::cout << "Running the INPUT_IMAGE -> GET_IMAGE example:" << std::endl;
  run_expecting_image_example();
  std::cin.get();

  return 0;
}


void run_single_image_example()
{
  // Only 20% of the available GPU memory will be allocated
  float gpu_memory_fraction = 0.2f;

  // the model will try to infer the input and output layer names automatically 
  // (only use if it's a simple "one-input -> one-output" model
  bool inferInputOutput = true;

  // load a model from a .pb file
  tf_image::TF_Model model1;

  model1.loadModel( "graph_1.pb", gpu_memory_fraction, inferInputOutput );

  // load input image
  cv::Mat image = cv::imread( "image.jpg", cv::IMREAD_UNCHANGED );

  // resize the image to fit the model's input:
  cv::resize( image, image, { 224,244 } );

  // run prediction:  
  std::vector< std::vector< float > > results = model1.predict<std::vector<float>>( { image } );
  //   ^              ^ second vector is a normal model output (i.e. for classification or regression)
  //   ^ the elements of the first vector correspond to the model's outputs (if the model has only one, the vector contains only 1 vector)

  // print results
  for ( size_t i = 0; i < results.size(); i++ )
  {
    std::cout << "Output vector #" << i << ": ";
    for ( size_t j = 0; j < results[i].size(); j++ )
    {
      std::cout << std::fixed << std::setprecision(4) << results[i][j] << "\t";
    }
    std::cout << std::endl;
  }
}

void run_expecting_image_example() {

  // Only 20% of the available GPU memory will be allocated
  float gpu_memory_fraction = 0.2f;

  // the model will try to infer the input and output layer names automatically 
  // (only use if it's a simple "one-input -> one-output" model
  bool inferInputOutput = true;

  // load a model from a .pb file
  tf_image::TF_Model model2;
  model2.loadModel( "graph_2.pb", gpu_memory_fraction, inferInputOutput );

  // load input image
  cv::Mat image = cv::imread( "image.jpg", cv::IMREAD_UNCHANGED );
  
  // run prediction
  std::vector<cv::Mat> result = model2.predict<cv::Mat>( { image } );
  // the output image is type float32, and it can also contain any number of channels (even more than 4)

  // we can try to visualize it like a heatmap:
  cv::Mat heatmap = createHeatmap( result[0] );
  cv::resize( heatmap, heatmap, image.size() );

  std::cout << "Showing the image" << std::endl;
  while ( cv::waitKey( 1 ) != 27 ) {
    cv::imshow( "original input image", image );
    cv::imshow( "output heatmap", heatmap );
  }
  cv::destroyAllWindows();
}

cv::Mat createHeatmap( const cv::Mat& heatmaps ) {
  cv::Mat hue_ch = cv::Mat::zeros( heatmaps.rows, heatmaps.cols, CV_8U );
  cv::Mat sat_ch = cv::Mat::ones( heatmaps.rows, heatmaps.cols, CV_8U ) * 255;
  cv::Mat val_ch = cv::Mat::ones( heatmaps.rows, heatmaps.cols, CV_8U ) * 255;

  for ( int i = 0; i < heatmaps.channels(); i++ ) {
    cv::Mat h_ch, h_ch_uint8;
    cv::extractChannel( heatmaps, h_ch, i );

    h_ch *= 180;

    h_ch.convertTo( h_ch_uint8, CV_8U );

    hue_ch |= h_ch_uint8;
  }

  cv::Mat prettyHeatmap;
  cv::merge( std::vector<cv::Mat> { hue_ch, sat_ch, val_ch }, prettyHeatmap );

  cv::cvtColor( prettyHeatmap, prettyHeatmap, cv::COLOR_HSV2RGB );

  return prettyHeatmap;
}