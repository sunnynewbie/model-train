# model-train

supported version 3.7.14


Mac support::


download python 3.7.14 from 
https://www.python.org/ftp/python/3.7.14/ 
tar.gz file 

follow below command for installation perticular version of python

cd
mkdir localpython
cd localpython
curl  https://www.python.org/ftp/python/{version}/Python-{version}.tgz > Python-{version}.tgz

tar -zxvf Python-2.7.15.tgz
cd Python-2.7.15

# "make clean" may be necessary here for earlier versions
./configure --prefix=${HOME}/localpython --enable-optimizations
make
make install

if you found zlib issue Go to python/Modues/ uncomment line 
"zlib zlibmodule.c -I$(prefix)/include -L$(exec_prefix)/lib -lz" to enable zlib installation with python version 3.7.14



you can use Environment or use version directly


pip install -q tflite-model-maker
