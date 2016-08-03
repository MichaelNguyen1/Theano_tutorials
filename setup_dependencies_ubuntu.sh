echo 'Installing dependencies for Theano...'

# Obtain basic tools
sudo apt-get install vim
sudo apt-get python-setuptools python-dev build-essential
sudo easy_install pip

# Obtain the python libraries
sudo pip install numpy
sudo pip install theano
