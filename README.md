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
CUDA-GDB is the NVIDIA tool for debugging CUDA applications running on Linux and QNX. To obtain a copy of the source code for cuda-gdb using the RPM and Debian installation methods, the cuda-gdb-src package must be installed. You may want to consult https://docs.nvidia.com/cuda/cuda-gdb/index.html for installing the package.

## Uninstalling CUDA and NVIDIA drivers
In case you wanted to do everything again from scratch, or have an older version installed, you better remove the existing packages and dependencies, and install the suitable versions from the beginning. You can run the following commands:
```
# To remove CUDA Toolkit:
$ sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
# To remove NVIDIA Drivers:
$ sudo apt-get --purge remove "*nvidia*"
```
## Installation of Docker engine
As was previously mentioned, we would like to install the Docker version of DSSTNE. Therefore, we need to install Docker Engine as well. You can find a detailed installation manual at https://docs.docker.com/engine/install/ubuntu/. In this section, however, we will continue with installation of the latest Docker Engine by far on Ubuntu 20.04 (same for 18.04).
### Uninstall old versions
If you happen to have some older version of Docker installed, uninstall it first.
```
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```
### Setup the repository
```
$ sudo apt-get update

$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
Search for the last 8 characters of the fingerprint, and verify that you now have the key with the fingerprint ```9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88``` by running the following command:
```
$ sudo apt-key fingerprint 0EBFCD88
```
### set up the stable repo
```
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```
### Install the Docker Engine
```
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```
### Verify the Docker Engine
```
$ sudo docker run hello-world
```
You should see something like this:

![Hello World by Docker](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2014-51-59.png)

## Installation of NVIDIA Container Toolkit
The NVIDIA Container Toolkit allows users to build and run GPU accelerated Docker containers. The toolkit includes a container runtime library and utilities to automatically configure containers to leverage NVIDIA GPUs. In oder to use the toolkit, we need to make sure that Docker Engine 19.03 or higher has been installed on the computer and NVIDIA drivers are up and running. If you have been following this guide since the beginning, then you are ready to install the NVIDIA Container Toolkit:
# Add the package repositories
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
### Running the toolkit
You can use any of the below situations as examples to run the container. Pay attention that the docker image used here is based on the image file seen on DSSTNE library. So, you need to skip here to the later section and come back afterwards to see how it does.
```
#### Test nvidia-smi with the latest official CUDA image
# Start a GPU enabled container on all GPUs
sudo docker run --gpus all nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 nvidia-smi

# Start a GPU enabled container on two GPUs
sudo docker run --gpus 2 nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 nvidia-smi

# Starting a GPU enabled container on specific GPUs
sudo docker run --gpus '"device=1,2"' nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 nvidia-smi
sudo docker run --gpus '"device=UUID-ABCDEF,1"' nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 nvidia-smi

# Specifying a capability (graphics, compute, ...) for my container
# Note this is rarely if ever used this way
sudo docker run --gpus all,capabilities=utility nvidia/cuda:10.0-base nvidia-smi
```
If you run it on all GPUs, you should see something like the following image:

![NVIDIA Docker run](https://github.com/omikay/dsstne-installation/blob/master/Images/Screenshot%20from%202020-06-30%2015-19-08.png)

## Build Amazon DSSTNE's Docker image
```
$ git clone https://github.com/amznlabs/amazon-dsstne.git
$ cd amazon-dsstne/
```
You need to make some changes to the Dockerfile here in order not to get any errors in the future. First open the Dockerfile:
```
$ sudo nano ./Dockerfile
```
Now you need to replace ```RUN apt-get install -y libnetcdf-c++4-dev``` with the following:
```
$ RUN wget http://archive.ubuntu.com/ubuntu/pool/universe/n/netcdf-cxx/netcdf-cxx_4.3.0+ds.orig.tar.gz && \
	  tar xzf netcdf-cxx_4.3.0+ds.orig.tar.gz && \
	  cd netcdf-cxx4-4.3.0 && \
    ./configure --disable-filter-testing && \
    make && \
    make install
```
and replace ```ENV PATH=/opt/amazon/dsstne/bin/:${PATH}``` with ```ENV PATH=/opt/amazon/dsstne/build/bin/:${PATH}```.

You can now build the image:
```
$ sudo docker build -t amazon-dsstne .
```
## Test the Docker image
```
$ sudo docker run --rm -it amazon-dsstne predict
```
The following error that is 'Missing required argument' error is expected to be seen after this test:
```
Error: Missing required argument: -d: dataset_name is not specified.
Predict: Generates predictions from a trained neural network given a signals/input dataset.
Usage: predict -d <dataset_name> -n <network_file> -r <input_text_file> -i <input_feature_index> -o <output_feature_index> -f <filters_json> [-b <batch_size>] [-k <num_recs>] [-l layer] [-s input_signals_index] [-p score_precision]
    -b batch_size: (default = 1024) the number records/input rows to process in a batch.
    -d dataset_name: (required) name for the dataset within the netcdf file.
    -f samples filterFileName .
    -i input_feature_index: (required) path to the feature index file, used to tranform input signals to correct input feature vector.
    -k num_recs: (default = 100) The number of predictions (sorted by score to generate). Ignored if -l flag is used.
    -l layer: (default = Output) the network layer to use for predictions. If specified, the raw scores for each node in the layer is output in order.
    -n network_file: (required) the trained neural network in NetCDF file.
    -o output_feature_index: (required) path to the feature index file, used to tranform the network output feature vector to appropriate features.
    -p score_precision: (default = 4.3f) precision of the scores in output
    -r input_text_file: (required) path to the file with input signal to use to generate predictions (i.e. recommendations).
    -s filename (required) . to put the output recs to.
```
If it all went well you need to start a new shell on a fresh Docker container. But you have to set the runtime to nvidia so that the docker file could communicate with your GPU.
```
$ sudo docker run -it --runtime=nvidia amazon-dsstne /bin/bash
```
## Run DSSTNE on some example datasets
```
# Fetch Movielens dataset
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

# Extract ratings from dataset
unzip -p ml-20m.zip ml-20m/ratings.csv > ml-20m_ratings.csv

# Convert ml-20m_ratings.csv to format supported by generateNetCDF
mv ml-20m_ratings.csv /opt/amazon/dsstne/samples/movielens
cd /opt/amazon/dsstne/samples/movielens
awk -f convert_ratings.awk ml-20m_ratings.csv > ml-20m_ratings

# Generate NetCDF files for input and output layers
generateNetCDF -d gl_input  -i ml-20m_ratings -o gl_input.nc  -f features_input  -s samples_input -c
generateNetCDF -d gl_output -i ml-20m_ratings -o gl_output.nc -f features_output -s samples_input -c

# Train the network
train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

# Generate predictions
predict -b 256 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml-20m_ratings -s recs -r ml-20m_ratings
```
Your predicitons will be stores in a folder named 'recs'.
