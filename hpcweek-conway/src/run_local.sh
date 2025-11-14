cd ..
# apt install python3-pybind11
# apt install python3-setuptools
# pip3 install -r requirements.txt
rm -f NG.cpython-311-x86_64-linux-gnu.so
rm -f src/NG.cpython-311-x86_64-linux-gnu.so
python3 setup.py build_ext --inplace
cp NG.cpython-311-x86_64-linux-gnu.so src/NG.cpython-311-x86_64-linux-gnu.so
cd src
python3 main.py -S 200 200 -I 200