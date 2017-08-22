# DetectNet_ROS
NVIDIA provides [deep-learning inference](https://github.com/dusty-nv/jetson-inference) networks and deep vision primitives with TensorRT and Jetson TX1/TX2. 'DetectNet' performs detecting objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes). 
  
ROS topic can be used as image input(Stereolab's ZED camera is used for implementing) for DetectNet using DetectNet_ROS.
  
# Pre-requisite
- Jetson TX2 with JetPack 3.1(R28.1)
- TensorRT 2.1
- CUDA 8.0
- cuDNN 6.1
- OpenCV 2.4.13
- jetson-inference
