rmdir /S /q zluda
curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm6-amd64.zip > zluda.zip
tar -xf zluda.zip
del zluda.zip
copy env\Lib\site-packages\torch\lib\nvrtc64_112_0.dll env\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y
copy zluda\cublas.dll env\Lib\site-packages\torch\lib\cublas64_11.dll /y
copy zluda\cusparse.dll env\Lib\site-packages\torch\lib\cusparse64_11.dll /y
copy zluda\nvrtc.dll env\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y
copy zluda\cufft.dll env\Lib\site-packages\torch\lib\cufft64_10.dll /y
copy zluda\cufftw.dll env\Lib\site-packages\torch\lib\cufftw64_10.dll /y
pause