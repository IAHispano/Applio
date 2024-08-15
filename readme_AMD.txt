1) Install VC++ Runtime - https://aka.ms/vs/17/release/vc_redist.x64.exe

2) Install HIP SDK
	read https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html  
	"Windows-supported GPUs" section
	
	if your GPU has a green checkbox in HIP SDK column use step a)
	if your GPU has red cross in HIP SDK column use step b)
	if your GPU is not listed use step c)
	
	a) install either v6.1.2 or v5.7.1 HIP SDK https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
	
	b) install v5.7.1 HIP SDK
		- 6700, 6700XT, 6750XT download https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1031.7z
		- 6600, 6600XT, 6650XT download https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1032.7z
		
		rename C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library to library.old
		create a new folder called 'library'
		unzip the content of the archive into that folder
		
	c) install v5.7.1 HIP SDK 
	    google "techpowerup your_gpu" to find the value of "Shader ISA" (gfxnnnn)
		only gfx803/900/906/1010/1011/1012/1030/1100/1101/1102 are supported by this option
		
		download https://github.com/brknsoul/ROCmLibs/raw/main/ROCmLibs.7z
	
		rename C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library to library.old
		unzip the content of the archive into C:\Program Files\AMD\ROCm\5.7\bin\rocblas\
	
3) Patching Applio
	
	a) If you have downloaded ApplioV3.2.1_compiled.zip and unzipped it already
		run reinstall-torch.bat
		
	b) If you have downloaded Applio-3.2.1_src.zip
		modify run-install.bat to change the torch versions (line 67?)
			> pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
		start run-install.bat to download all the required libraries
		
	apply the code changes from the pull request manually
	
		
4) Download Zluda and patch torch libraries
	
	a) if you have HIP SDK 5.7 installed, 
		use patch_zluda_hip57.bat
		make sure "C:\Program Files\AMD\ROCm\5.7\bin" is added to Path in environment variables
		
	b) if you have HIP SDK 6.1 installed, 
		use patch_zluda_hip61.bat
		make sure "C:\Program Files\AMD\ROCm\6.1\bin" is added to Path in environment variables
	
5) Use run-applio-zluda.bat to start Applio