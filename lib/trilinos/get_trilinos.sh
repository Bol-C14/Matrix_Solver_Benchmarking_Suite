#!/bin/bash

# Update and install prerequisite packages
sudo apt-get update
sudo apt-get install -y gcc g++ gfortran make cmake bison flex libfl-dev libfftw3-dev \
libsuitesparse-dev libblas-dev liblapack-dev libtool autoconf automake git libopenmpi-dev openmpi-bin

# Set up directories and download Trilinos 12.12.1
TRILINOS_VERSION="trilinos-release-12-12-1"
TRILINOS_DOWNLOAD_DIR="$HOME/Downloads"
TRILINOS_INSTALL_DIR="$HOME/Trilinos12.12"

mkdir -p $TRILINOS_DOWNLOAD_DIR
mkdir -p $TRILINOS_INSTALL_DIR

cd $TRILINOS_DOWNLOAD_DIR
wget https://github.com/trilinos/Trilinos/archive/refs/tags/$TRILINOS_VERSION.tar.gz

# Extract Trilinos
cd $TRILINOS_INSTALL_DIR
tar xzf $TRILINOS_DOWNLOAD_DIR/$TRILINOS_VERSION.tar.gz

echo "Prerequisites installed and Trilinos downloaded."
