if [ ! -d raw_voice_cleanup ]; then
   git clone https://github.com/kastnerkyle/raw_voice_cleanup
fi

# Setup tested on AWS AMI
# Deep Learning AMI (Ubuntu) Version 21.2 - ami-0a47106e391391252

sudo apt-get update
sudo apt-get install build-essential software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install gcc-6 g++-6 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo apt-get install gfortran-6 libgfortran-6-dev  -y
sudo apt-get install automake autoconf subversion libatlas3-base -y
sudo apt-get install ffmpeg
# is sh is linked to dash a bunch of kaldi build breaks? oof
sudo ln -s -f bash /bin/sh

gcc -v
mkdir -p raw_wav
mkdir -p raw_json

cd raw_voice_cleanup/alignment
# a lot of the tools only work with python 2
ln -s /home/ubuntu/anaconda2 /home/ubuntu/anaconda
export PATH="/home/ubuntu/anaconda/bin:$PATH"
bash -x build_gentle.sh

# a lot of copying of files for
# OpenFST bitrot https://github.com/hfst/hfst/issues/358
# replace
#      (fst_.GetImpl())->template CacheImpl<A>::InitArcIterator(state_,#
# with 
#       (fst_.GetImpl())->CacheImpl<A>::InitArcIterator(state_
# also use g++-6
