#!/bin/bash

TOOL_NAME=venus2
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(realpath $0))

./xphostid
apt update
apt upgrade

apt install nvidia-driver-515 nvidia-dkms-515
nvidia-smi

apt -y install software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt install -y python3.7
apt install -y python3-pip
apt install -y python3.7-distutils
apt install -y python3.7-dev
apt install -y psmisc # for killall, used in prepare_instance.sh script


pip3 install pipenv
pipenv install --skip-lock
pipenv run pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pipenv install onnx2torch --skip-lock
pipenv run python -c "import torch; print(torch.cuda.is_available()); print(torch.zeros((1,)).cuda())"

#wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
#tar -xzvf gurobi9.1.2_linux64.tar.gz 
#rm gurobi9.1.2_linux64.tar.gz 
#mv gurobi912 "$DIR/"
#cd "$DIR/gurobi912/linux64/"
#sudo python3 setup.py install

#  Gurobi license: uncomment below and replace xxx with the license key.
#cd bin
#./grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#cd ../../../

