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
The kernel headers and development packages for the currently running kernel can be installed with:
```sh
$ sudo apt-get install linux-headers-$(uname -r)
```
## Download and install the NVIDIA CUDA Toolkit
The NVIDIA CUDA Toolkit is available at http://developer.nvidia.com/cuda-downloads. The following is what you need to choose to download the right file for this version of Ubuntu:
![CUDA Toolkit version](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2012-12-39.png)

The following steps should be run in the commandline to download and install the toolkit on the host machine. The version of CUDA to be installed will be CUDA v11.

```sh
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.1-450.36.06-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.1-450.36.06-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda
```
## Environment Setup
```sh
$ export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
```

## POWER9 setup
The NVIDIA Persistence Daemon should be automatically started for POWER9 installations. Check that it is running with the following command:
```sh
$ systemctl status nvidia-persistenced
```
If it is not active, run the following command:
```sh
$ sudo systemctl enable nvidia-persistenced
```
The udev rule must be disabled in order for the NVIDIA CUDA driver to function properly on POWER9 systems. Run the following to see the rules:
```sh
$ sudo nano /lib/udev/rules.d/40-vm-hotadd.rules
```
The rule would look like something like this:
![Hotadd Rule](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2012-34-31.png)

The part that we are interested in is:
```sh
# Memory hotadd request
SUBSYSTEM=="memory", ACTION=="add", DEVPATH=="/devices/system/memory/memory[0-9]*", TEST=="state", ATTR{state}="online"
```
This rule must be disabled by copying the file to ```sh/etc/udev/rules.d``` and commenting out, removing, or changing the hot-pluggable memory rule in the ```sh/etc``` copy so that it does not apply to POWER9 NVIDIA systems.
```sh
# to copy
$ sudo cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d
# to modify
sudo sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules
```
You will need to reboot the system to initialize the above changes.

