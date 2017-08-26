#include <ros/ros.h>
#include <jetson-inference/detectNet.h>
#include <jetson-inference/loadImage.h>
#include <jetson-inference/cudaFont.h>
#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/cudaNormalize.h>
#include <jetson-inference/cudaFont.h>

#include <jetson-inference/glDisplay.h>
#include <jetson-inference/glTexture.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>

bool signal_recieved;

std_msgs::Int32 number_detected;
std_msgs::Int32MultiArray array_x_min;
std_msgs::Int32MultiArray array_y_min;
std_msgs::Int32MultiArray array_x_max;
std_msgs::Int32MultiArray array_y_max;

ros::Publisher detect_num;
ros::Publisher detect_x_min;
ros::Publisher detect_y_min;
ros::Publisher detect_x_max;
ros::Publisher detect_y_max;

cv::Mat cv_image;
sensor_msgs::ImagePtr pub_detectnet_image;

detectNet* net;

float4* gpu_data;

uint32_t imgWidth;
uint32_t imgHeight;
size_t   imgSize;

float* bbCPU;
float* bbCUDA;
float* confCPU;
float* confCUDA;

uint32_t maxBoxes;
uint32_t classes;

void callback(const sensor_msgs::ImageConstPtr& input)
{
  cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image; 
  cv_image = cv_im;    

  // convert bit depth
  cv_im.convertTo(cv_im,CV_32FC3);
  
  // convert color
  cv::cvtColor(cv_im,cv_im,CV_BGR2RGBA);  

  // allocate GPU data if necessary
  if(!gpu_data)
  {
    ROS_INFO("first allocation");
    CUDA(cudaMalloc(&gpu_data, cv_im.rows*cv_im.cols * sizeof(float4)));
  }
  else if(imgHeight != cv_im.rows || imgWidth != cv_im.cols)
  {
    ROS_INFO("re allocation");
    
    // reallocate for a new image size if necessary
    CUDA(cudaFree(gpu_data));
    CUDA(cudaMalloc(&gpu_data, cv_im.rows*cv_im.cols * sizeof(float4)));
  }

  imgHeight = cv_im.rows;
  imgWidth = cv_im.cols;
  imgSize = cv_im.rows*cv_im.cols * sizeof(float4);
  float4* cpu_data = (float4*)(cv_im.data);

  // copy to device
  CUDA(cudaMemcpy(gpu_data, cpu_data, imgSize, cudaMemcpyHostToDevice));

  float confidence = 0.0f;

  // detect image with detectNet
  int numBoundingBoxes = maxBoxes;

  if( net->Detect((float*)gpu_data, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU))
  {
    int lastClass = 0;
    int lastStart = 0;

    //printf("%i bounding boxes detected\n", numBoundingBoxes);

    // number of bounding boxes   
    number_detected.data = numBoundingBoxes;			

    // initialize number array of bounding box pixels
    array_x_min.data.clear();
    array_y_min.data.clear();
    array_x_max.data.clear();
    array_y_max.data.clear();

    for( int n=0; n < numBoundingBoxes; n++ )
    {
	const int nc = confCPU[n*2+1];
	float* bb = bbCPU + (n * 4);
		
	printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
		
	array_x_min.data.push_back(bb[0]);
	array_y_min.data.push_back(bb[1]);
	array_x_max.data.push_back(bb[2]);
	array_y_max.data.push_back(bb[3]);	

        // draw a green line(CW) on the overlay copy
	cv::line(cv_image, cv::Point(bb[0],bb[1]), cv::Point(bb[2],bb[1]),cv::Scalar(0, 255, 0),2);
	cv::line(cv_image, cv::Point(bb[2],bb[1]), cv::Point(bb[2],bb[3]),cv::Scalar(0, 255, 0),2);
	cv::line(cv_image, cv::Point(bb[2],bb[3]), cv::Point(bb[0],bb[3]),cv::Scalar(0, 255, 0),2);
	cv::line(cv_image, cv::Point(bb[0],bb[3]), cv::Point(bb[0],bb[1]),cv::Scalar(0, 255, 0),2);

	if( nc != lastClass || n == (numBoundingBoxes - 1) )
	{
	    if( !net->DrawBoxes((float*)gpu_data, (float*)gpu_data, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
		printf("detectnet-console:  failed to draw boxes\n");
				
	    lastClass = nc;
	    lastStart = n;

	    CUDA(cudaDeviceSynchronize());
	}
    }

  pub_detectnet_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image).toImageMsg();						
}


}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ros_detectnet_publisher");
  ros::NodeHandle nh;
  detectNet::NetworkType networkType;
  std::string detectnet_name, sub_ros_topic;
  std::string prototxt_path, model_path, mean_binary_path;

  net = NULL;

  gpu_data = NULL;

  imgWidth = 0;
  imgHeight = 0;
  imgSize = 0;

  bbCPU = NULL;
  bbCUDA = NULL;
  confCPU = NULL;
  confCUDA = NULL;

  maxBoxes = 0;
  classes = 0;

  if(nh.getParam("/type", detectnet_name))
  {
    if ( detectnet_name == "pednet_multi" ){
        networkType = detectNet::PEDNET_MULTI;
        prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/multiped-500/deploy.prototxt";
        model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/multiped-500/snapshot_iter_178000.caffemodel";
        mean_binary_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/multiped-500/mean.binaryproto";
    }
    else if ( detectnet_name == "facenet" ){
        networkType = detectNet::FACENET;
        prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/facenet-120/deploy.prototxt";
    model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/facenet-120/snapshot_iter_24000.caffemodel";
    mean_binary_path = "";
    }
    else if ( detectnet_name == "pednet" ){
        networkType = detectNet::PEDNET;
        prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/ped-100/deploy.prototxt";
        model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/ped-100/snapshot_iter_70800.caffemodel";
        mean_binary_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/ped-100/mean.binaryproto";
    }
    else if ( detectnet_name == "coco-airplane" ){
        networkType = detectNet::COCO_AIRPLANE;
        prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Airplane/deploy.prototxt";
        model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Airplane/snapshot_iter_22500.caffemodel";
        mean_binary_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Airplane/mean.binaryproto";
    }
    else if ( detectnet_name == "coco-bottle" ){
        networkType = detectNet::COCO_BOTTLE;
	prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Bottle/deploy.prototxt";
        model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Bottle/snapshot_iter_59700.caffemodel";
        mean_binary_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Bottle/mean.binaryproto";
    }
    else if ( detectnet_name == "coco-chair" ){
        networkType = detectNet::COCO_CHAIR;
	prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Chair/deploy.prototxt";
        model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Chair/snapshot_iter_89500.caffemodel";
        mean_binary_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Chair/mean.binaryproto";
    }
    else if ( detectnet_name == "coco-dog" ){
        networkType = detectNet::COCO_DOG;
	prototxt_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Dog/deploy.prototxt";
        model_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Dog/snapshot_iter_38600.caffemodel";
        mean_binary_path = "/home/nvidia/jetson-inference/build/aarch64/bin/networks/DetectNet-COCO-Dog/mean.binaryproto";
    }
    else
        ROS_ERROR("failed to get detectnet type");
  }
  else
    ROS_ERROR("failed to get detectnet type");

  // make sure files exist (and we can read them)
  if( access(prototxt_path.c_str(),R_OK) )
    ROS_ERROR("unable to read file \"%s\", check filename and permissions",prototxt_path.c_str());
  if( access(model_path.c_str(),R_OK) )
    ROS_ERROR("unable to read file \"%s\", check filename and permissions",model_path.c_str());
  if( networkType != detectNet::FACENET && access(mean_binary_path.c_str(),R_OK) )
    ROS_ERROR("unable to read file \"%s\", check filename and permissions",mean_binary_path.c_str());

  if(!nh.getParam("/ros_topic", sub_ros_topic))
  {
    ROS_ERROR("failed to get topic");
  }

  //create detectNet
  if ( networkType == detectNet::FACENET )
    net = detectNet::Create(prototxt_path.c_str(),model_path.c_str(),NULL);
  else
    net = detectNet::Create(prototxt_path.c_str(),model_path.c_str(),mean_binary_path.c_str());
  
  if( !net )
  {
    ROS_ERROR("ros-detectnet-camera: failed to create detectNet\n");
  }

  // allocate memory for output bounding boxes and class confidence
  maxBoxes = net->GetMaxBoundingBoxes();		
  printf("maximum bounding boxes:  %u\n", maxBoxes);

  classes  = net->GetNumClasses();	

  if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
  {
    ROS_ERROR("detectnet-console:  failed to alloc output memory\n");
  }

  // setup image transport
  image_transport::ImageTransport it(nh);
              
  // subscriber for passing in images
  image_transport::Subscriber sub = it.subscribe(sub_ros_topic, 10, callback);
  
  // publisher for output image
  image_transport::Publisher pub = it.advertise("detectnet/image", 1);
  	
  // publisher for number of detected bounding boxes output       
  ros::Publisher pub_detect_num = nh.advertise<std_msgs::Int32>("detectNet/number",10);

  // publish for detect region output
  ros::Publisher pub_detect_x_min = nh.advertise<std_msgs::Int32MultiArray>("detectNet/x_min", 30);
  ros::Publisher pub_detect_y_min = nh.advertise<std_msgs::Int32MultiArray>("detectNet/y_min", 30);
  ros::Publisher pub_detect_x_max = nh.advertise<std_msgs::Int32MultiArray>("detectNet/x_max", 30);
  ros::Publisher pub_detect_y_max = nh.advertise<std_msgs::Int32MultiArray>("detectNet/y_max", 30);

  ros::Rate loop_rate(25); //Hz

  while (nh.ok()) {
    pub.publish(pub_detectnet_image);

    pub_detect_num.publish(number_detected);	
    pub_detect_x_min.publish(array_x_min);
    pub_detect_y_min.publish(array_y_min);
    pub_detect_x_max.publish(array_x_max);
    pub_detect_y_max.publish(array_y_max);

    ros::spinOnce();
    loop_rate.sleep();
  }
}
