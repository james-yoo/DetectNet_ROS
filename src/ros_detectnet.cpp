#include <ros/ros.h>
#include <jetson-inference/detectNet.h>

#include <jetson-inference/loadImage.h>
#include <jetson-inference/cudaFont.h>

#include <jetson-inference/glDisplay.h>
#include <jetson-inference/glTexture.h>

#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/cudaNormalize.h>
#include <jetson-inference/cudaFont.h>

#include <opencv2/core.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>

#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

namespace ros_detectnet_zed{

class ros_detectnet : public nodelet::Nodelet
{
    public:
        ~ros_detectnet()
        {
            ROS_INFO("\nshutting down...\n");
            if(gpu_data)
                CUDA(cudaFree(gpu_data));
            delete net;
        }
        void onInit()
        {
            // get a private nodehandle
            ros::NodeHandle& private_nh = getPrivateNodeHandle();

	    bbCPU    = NULL;
	    bbCUDA   = NULL;
	    confCPU  = NULL;
	    confCUDA = NULL;

            // get parameters from server, checking for errors as it goes
            std::string prototxt_path, model_path, mean_binary_path;
            if(! private_nh.getParam("prototxt_path", prototxt_path) )
                ROS_ERROR("unable to read prototxt_path for detectnet node");
            if(! private_nh.getParam("model_path", model_path) )
                ROS_ERROR("unable to read model_path for detectnet node");
            if(! private_nh.getParam("mean_binary_path", mean_binary_path) )
                ROS_ERROR("unable to read mean_binary_path for detectnet node");

            // make sure files exist (and we can read them)
            if( access(prototxt_path.c_str(),R_OK) )
                 ROS_ERROR("unable to read file \"%s\", check filename and permissions",prototxt_path.c_str());
            if( access(model_path.c_str(),R_OK) )
                 ROS_ERROR("unable to read file \"%s\", check filename and permissions",model_path.c_str());
            if( access(mean_binary_path.c_str(),R_OK) )
                 ROS_ERROR("unable to read file \"%s\", check filename and permissions",mean_binary_path.c_str());

            // create detectNet
	    net = detectNet::Create(prototxt_path.c_str(),model_path.c_str(),mean_binary_path.c_str());

            if( !net )
            {
                ROS_INFO("detectnet-console:   failed to initialize detectNet\n");
                return;
            }

	/*
	 * allocate memory for output bounding boxes and class confidence
	 */
	maxBoxes = net->GetMaxBoundingBoxes();		
	printf("maximum bounding boxes:  %u\n", maxBoxes);
	
	classes  = net->GetNumClasses();	

	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
	}

            // setup image transport
            image_transport::ImageTransport it(private_nh);
            
	    // subscriber for passing in images
            imsub = it.subscribe("imin", 10, &ros_detectnet::callback, this);
    	
	    // publisher for number of detected bounding boxes output       
            detect_num = private_nh.advertise<std_msgs::Int32>("detectNet/number",10);

	    // publish for detect region output
	    detect_x_min = private_nh.advertise<std_msgs::Int32MultiArray>("detectNet/x_min", 30);
	    detect_y_min = private_nh.advertise<std_msgs::Int32MultiArray>("detectNet/y_min", 30);
	    detect_x_max = private_nh.advertise<std_msgs::Int32MultiArray>("detectNet/x_max", 30);
	    detect_y_max = private_nh.advertise<std_msgs::Int32MultiArray>("detectNet/y_max", 30);

            // init gpu memory
            gpu_data = NULL;
        }


    private:
        void callback(const sensor_msgs::ImageConstPtr& input)
        {
            cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image;
            ROS_INFO("image ptr at %p",cv_im.data);
            // convert bit depth
            cv_im.convertTo(cv_im,CV_32FC3);
            // convert color
            cv::cvtColor(cv_im,cv_im,CV_BGR2RGBA);

            // allocate GPU data if necessary
            if(!gpu_data){
                ROS_INFO("first allocation");
                CUDA(cudaMalloc(&gpu_data, cv_im.rows*cv_im.cols * sizeof(float4)));
            }else if(imgHeight != cv_im.rows || imgWidth != cv_im.cols){
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

            // classify image
            //const int img_class = net->Classify((float*)gpu_data, imgWidth, imgHeight, &confidence);

	// detect image with detectNet
	int numBoundingBoxes = maxBoxes;
	
		if( net->Detect((float*)gpu_data, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);
		
			// publish number of bounding boxes					
			std_msgs::Int32 number_detectNet;
            		number_detectNet.data = numBoundingBoxes;			

			int lastClass = 0;
			int lastStart = 0;
			
			// publish number array of bounding box pixels					
			std_msgs::Int32MultiArray array_x_min;
			std_msgs::Int32MultiArray array_y_min;
			std_msgs::Int32MultiArray array_x_max;
			std_msgs::Int32MultiArray array_y_max;

			array_x_min.data.clear();
			array_y_min.data.clear();
			array_x_max.data.clear();
			array_y_max.data.clear();

			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);
				
				//printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
				
				array_x_min.data.push_back(bb[0]);
				array_y_min.data.push_back(bb[1]);
				array_x_max.data.push_back(bb[2]);
				array_y_max.data.push_back(bb[3]);
				
				if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes((float*)gpu_data, (float*)gpu_data, imgWidth, imgHeight, 
						                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet-console:  failed to draw boxes\n");
						
					lastClass = nc;
					lastStart = n;

					CUDA(cudaDeviceSynchronize());
				}
			}
			
			detect_num.publish(number_detectNet);
			
			detect_x_min.publish(array_x_min);
			detect_y_min.publish(array_y_min);
			detect_x_max.publish(array_x_max);
			detect_y_max.publish(array_y_max);
			
			
		}

        }

        // private variables
        image_transport::Subscriber imsub;
	       
        ros::Publisher detect_num;
        ros::Publisher detect_x_min;
        ros::Publisher detect_y_min;
        ros::Publisher detect_x_max;
        ros::Publisher detect_y_max;

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
};

PLUGINLIB_DECLARE_CLASS(ros_detectnet_zed, ros_detectnet, ros_detectnet_zed::ros_detectnet, nodelet::Nodelet);

} // namespace ros_deep_learning
