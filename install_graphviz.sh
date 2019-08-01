#!/bin/bash

# update system

sudo apt-get update

# install dependencies

sudo apt install -y libgd-dev fontconfig libcairo2-dev libpango1.0-dev libgts-dev

# download package

wget --directory-prefix=/tmp  http://archive.ubuntu.com/ubuntu/pool/universe/g/graphviz/graphviz_2.40.1.orig.tar.gz 
tar -xvf /tmp/graphviz_2.40.1.orig.tar.gz -C /tmp/

bash -c "cd /tmp/graphviz-2.40.1; ./configure; make; make install"


