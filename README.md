# Installation of Amazon DSSTNE from scratch
Amazon DSSTNE (Deep Scalable Sparse Tensor Network Engine) is a software library open sourced by Amazon in 2016 to train and deploy recommender systems. Despite deep learning algorithms have almost conquered all the fields that machine learning has entered, DSSTNE still performs quite well compared to deep learning models built on TensorFlow. The reason is the architecture of DSSTNE that enables it to work well with highly sparse data, which is the case when it comes to building recommender systems, and also the fact that it can run on multiple GPUs, which could significantly increase the training speed. The last point about this engine is its capability of scaling up when necessary.

DSSTNE can be at production level for deployment of real-world applications. Its architecture is comprised of sparse input neurons, fully connected hidden layers, and sparse output neurons.

In what follows, you can find a thoroughly illuststrated path towards installation of the required packages on the system before installing DSSTNE, followed by DSSTNE installation, and finally testing the installation with training on some sample datasets.

You can find the introductory part of the library on this repo page:
https://github.com/amzn/amazon-dsstne

## NVIDIA Cuda Installation Guide for Ubuntu 18.04 / 20.04
Training neural networks on CPUs is not such a good idea, since the number of calculations is quite high, and it would take a lot of time. Now, since many of these calculations are independent of each other, and each calculation instance is actually quite simple (summation, subtraction, function output calculation, etc.), we conclude that we do not need much of dedicated processing capacity for each one. Here is where using GPUs comes into play. The number of cores on a single GPU is something roughly between 4 to 30 times the number of cores on a CPU that you own (the current CPUs have between 2 to 32 core), and despite the processing capacity of each core is not as much as it is for the cores on a CPU, it does not impose any problems due to simplicity of each calculation instance, all of which making parallel processing on GPUs a real problem solver.

Having said all that, we need to have some NVIDIA GPU harward installed on our computer device, then we need to have a driver called CUDA installed on the host computer that enables that parallel processing we look forward to. By installation of CUDA toolkit, we will have all the required dependencies and NVIDIA drivers to perform parallel processing on our GPU unit.

You can find a general CUDA installation guide for all Linux operating systems here:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

What we will do, however, is presenting the installation process of CUDA on Ubuntu 18.04 / 20.04 (ours was 20.04).

To use CUDA on your system, you will need the following installed:
- CUDA-capable GPU
- A supported version of Linux with a gcc compiler and toolchain
- NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)

The system we are using passes the first two conditions. However, for the sake of thoroughness, let us do the verification process as follows:

(In case of not seeing what is stated here, please go to the installation guide provided above to see how you could solve the problem)

## Verify You Have a CUDA-Capable GPU
To verify that your GPU is CUDA-capable, run the following in the command line:

```sh
$ lspci | grep -i nvidia
```
You need to see something like the following:
![CUDA-Capable GPU](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-19-23.png)

## Verify You Have a Supported Version of Linux
```sh
$ uname -m && cat /etc/*release
```
You need to see something like the following:
![Supported Version of Linux](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-43-28.png)

## Verify gcc is installed
Run the following in the command line:
```sh
$ gcc --version
```
You need to see something like the following:
![System Has gcc Installed](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-47-55.png)

## Verify the System has the Correct Kernel Headers and Development Packages Installed
The version of the kernel your system is running can be found by running the following command:
```sh
$ gcc --version
```
You need to see something like the following:
![System Has gcc Installed](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-47-55.png)
