#!/bin/bash

echo "Start installing prerequisite packages...... Access to internet is required."

# Update the package list
sudo apt-get update

# Install required packages for building Xyce
sudo apt-get install -y gcc g++ gfortran make cmake bison flex libfl-dev libfftw3-dev libsuitesparse-dev libblas-dev liblapack-dev libtool

# Install additional packages if building from GitHub repository
sudo apt-get install -y autoconf automake git

# Install packages for building the parallel version of Xyce
sudo apt-get install -y libopenmpi-dev openmpi-bin

# Clean up
sudo apt-get clean


echo "--------------------------------------------"
echo "All required packages for building Xyce have been installed successfully."
echo "--------------------------------------------"
echo "--------------- Installation End -----------"
echo "--------------------------------------------"
