#!/bin/bash

# Update the package list
sudo apt-get update

# Install required packages for building Xyce
sudo apt-get install -y gcc g++ gfortran make cmake bison flex libfl-dev libfftw3-dev libsuitesparse-dev libblas-dev liblapack-dev libtool

# Install additional packages if building from GitHub repository
sudo apt-get install -y autoconf automake git

# Check Ubuntu version
VERSION=$(lsb_release -rs)

if [[ "$VERSION" == "18.04" ]]; then
    echo "Detected Ubuntu 18.04 or older. Rebuilding OpenMPI without --enable-heterogeneous."

    # Uninstall the problematic OpenMPI package if installed
    sudo apt-get remove -y libopenmpi-dev openmpi-bin

    # Install dependencies for building OpenMPI
    sudo apt-get install -y dpkg-dev debhelper fakeroot gfortran

    # Create a directory to work in
    mkdir ~/openmpi-build && cd ~/openmpi-build

    # Download OpenMPI source package
    sudo apt-get source openmpi

    # Download build dependencies
    sudo apt-get build-dep openmpi

    # Navigate to the source directory
    cd openmpi-*

    # Modify the OpenMPI configuration to remove --enable-heterogeneous
    sed -i 's/--enable-heterogeneous//' debian/rules

    # Build the OpenMPI package without --enable-heterogeneous
    dpkg-buildpackage -rfakeroot -uc -b

    # Install the newly built OpenMPI package
    sudo dpkg -i ../*.deb

    echo "OpenMPI has been rebuilt and installed without --enable-heterogeneous."
else
    # Install packages for building the parallel version of Xyce on non-affected systems
    sudo apt-get install -y libopenmpi-dev openmpi-bin
    echo "OpenMPI installed from repositories."
fi

# Clean up
sudo apt-get clean

echo "All required packages for building Xyce have been installed successfully."
