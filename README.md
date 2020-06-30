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

### Verify You Have a CUDA-Capable GPU
To verify that your GPU is CUDA-capable, run the following in the command line:

```
$ lspci | grep -i nvidia
```
You need to see something like the following:

![CUDA-Capable GPU](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-19-23.png)

### Verify You Have a Supported Version of Linux
```
$ uname -m && cat /etc/*release
```
You need to see something like the following:

![Supported Version of Linux](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-43-28.png)

### Verify gcc is installed
Run the following in the command line:
```
$ gcc --version
```
You need to see something like the following:

![System Has gcc Installed](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2011-47-55.png)

### Verify the System has the Correct Kernel Headers and Development Packages Installed
The kernel headers and development packages for the currently running kernel can be installed with:
```
$ sudo apt-get install linux-headers-$(uname -r)
```
### Download and install the NVIDIA CUDA Toolkit
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
Reboot the system to load the NVIDIA drivers.
### Environment Setup
```
$ export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### POWER9 setup
The NVIDIA Persistence Daemon should be automatically started for POWER9 installations. Check that it is running with the following command:
```
$ systemctl status nvidia-persistenced
```
If it is not active, run the following command:
```
$ sudo systemctl enable nvidia-persistenced
```
The udev rule must be disabled in order for the NVIDIA CUDA driver to function properly on POWER9 systems. Run the following to see the rules:
```
$ sudo nano /lib/udev/rules.d/40-vm-hotadd.rules
```
The rule would look like something like this:

![Hotadd Rule](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2012-34-31.png)

The part that we are interested in is:
```
# Memory hotadd request
SUBSYSTEM=="memory", ACTION=="add", DEVPATH=="/devices/system/memory/memory[0-9]*", TEST=="state", ATTR{state}="online"
```
This rule must be disabled by copying the file to ```/etc/udev/rules.d``` and commenting out, removing, or changing the hot-pluggable memory rule in the ```/etc``` copy so that it does not apply to POWER9 NVIDIA systems.
```
# to copy
$ sudo cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d
# to modify
sudo sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules
```
### A writable copy of the samples
Install a writable copy of the samples then build and run the nbody sample:
```
$ cuda-install-samples-11.0.sh ~
$ cd ~/NVIDIA_CUDA-11.0_Samples/5_Simulations/nbody
$ make
$ ./nbody
```
You will need to reboot the system to initialize the above changes.

### Verify Driver Installation
If you installed the driver, verify that the correct version of it is loaded. When the driver is loaded, the driver version can be found by executing the command:
```
$ cat /proc/driver/nvidia/version
```
### Checking the CUDA Toolkit version
```
$ nvcc --version
```
### Compiling the Examples
The NVIDIA CUDA Toolkit includes sample programs in source form. You should compile them by changing to ```~/NVIDIA_CUDA-11.0_Samples``` and typing ```make```. The resulting binaries will be placed under ```~/NVIDIA_CUDA-11.0_Samples/bin```.
### Running the binaries
After compilation, find and run ```deviceQuery``` under ```~/NVIDIA_CUDA-11.0_Samples```. For our installations run the following in the commandline:
```
$ ./NVIDIA_CUDA-11.0_Samples/bin/x86_64/linux/release/deviceQuery
```
If the CUDA software is installed and configured correctly, the output for ```deviceQuery``` should look similar to that shown in the figure below:

![Running Binaries](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2013-11-05.png)

Running the ```bandwidthTest``` program ensures that the system and the CUDA-capable device are able to communicate correctly.
```
$ ./NVIDIA_CUDA-11.0_Samples/bin/x86_64/linux/release/deviceQuery
```
Its output looks like something like this:

![Bandwidth Test](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2013-15-05.png)

### Install Nsight Eclipse Plugins
First you need to download and install the Eclipse IDE for C/C++ developers. You can run the following in the command line:
```
$ sudo apt-get update
# The link below needs to be updated, so you can check the latest version here: https://www.eclipse.org/downloads/packages/, and make sure to download Eclipse IDE for C/C++ Developers
$ wget http://eclipse.mirror.rafal.ca/technology/epp/downloads/release/2020-06/R/eclipse-cpp-2020-06-R-linux-gtk-x86_64.tar.gz
# The filename might need to be updated
$ sudo tar xzvf eclipse-cpp-2020-06-R-linux-gtk-x86_64.tar.gz
```
To install the Nsight Eclipse Plugins, run the following command:
```
$ /usr/local/cuda-11.0/bin/nsight_ee_plugins_manage.sh install <eclipse_dir> 
```
Note that ```<eclipse_dir>``` is where you have you 'eclipse' directory, which is where that 'eclipse-cpp-2020-06-R-linux-gtk-x86_64.tar.gz' file was extracted.
### Install third-party libraries
```
$ sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```
### Install the source code for cuda-gdb (optional - We didn't do it!)
CUDA-GDB is the NVIDIA tool for debugging CUDA applications running on Linux and QNX. You may wanna consult https://docs.nvidia.com/cuda/cuda-gdb/index.html for installing the package.

