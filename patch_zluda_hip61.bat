rmdir /S /q zluda
curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.86cdab3b14b556e95eafe370b8e8a1a80e8d093b/ZLUDA-windows-amd64.zip > zluda.zip
tar -xf zluda.zip
del zluda.zip
copy zluda\cublas.dll env\Lib\site-packages\torch\lib\cublas64_11.dll /y
copy zluda\cusparse.dll env\Lib\site-packages\torch\lib\cusparse64_11.dll /y
copy zluda\nvrtc.dll env\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y
