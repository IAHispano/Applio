
## Installation and Setup Instructions

Zluda is a CUDA emulator that supports a select number of modern AMD GPUs. The following guide is for Windows installation of Zluda.

### 1. Install VC++ Runtime

Download and install the VC++ Runtime from [this link](https://aka.ms/vs/17/release/vc_redist.x64.exe).
 

### 2. Install HIP SDK

Read the [System Requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html)

Check the *"Windows-supported GPUs"* section to determine the correct installation steps:

2.1 If your GPU has a green checkbox in the HIP SDK column:

- Install either v6.1.2 or v5.7.1 HIP SDK from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)

2.2 If your GPU is RX 6600, 6600XT, 6650XT, 6700, 6700XT, 6750XT:
- Install v5.7.1 HIP SDK from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- For 6700, 6700XT, 6750XT, download [gfx1031 archive](https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1031.7z)
- For 6600, 6600XT, 6650XT, download [gfx1032 archive](https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1032.7z)
**Steps:**
a. Rename `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` to `library.old`
b. Create a new folder named `library`
c. Unzip the content of the archive into that folder

2.3 For all other AMD GPUs: find gfxNNNN value for your GPU by googling "techpowerup your_gpu" (listed under "Shader ISA" on the page).

2.3.1 For `gfx803, gfx900, gfx906, gfx1010, gfx1011, gfx1012, gfx1030, gfx1100, gfx1101, gfx1102` GPUs:
- Install v5.7.1 HIP SDK from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- Download [this archive](https://github.com/brknsoul/ROCmLibs/raw/main/ROCmLibs.7z)
	**Steps:**
	a. Rename `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` to `library.old`
	b. Unzip the content of the archive into `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\`

2.3.2 Other GPUs
- Visit [this repository with a collection of tensile libraries](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)
- Follow the description there.

### 3. Installing Applio
3.1 Download a compiled version of Applio v3.2.5 or higher, unzip to the desired folder.
3.3 Using command line from the Applio folder run
		```env\python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --upgrade --index-url https://download.pytorch.org/whl/cu118```

### 4. Download Zluda and Patch Torch Libraries
4.1 move all .bat files from `assets\zluda`to root Applio folder
4.2 For HIP SDK 5.7:
- Run `patch_zluda_hip57.bat`.
- Add `C:\Program Files\AMD\ROCm\5.7\bin` to your system's Path environment variable.

4.3 For HIP SDK 6.1:**
- Run `patch_zluda_hip61.bat`.
- Add `C:\Program Files\AMD\ROCm\6.1\bin` to your system's Path environment variable.

### 5. Starting Applio

It is assumed your primary AMD GPU has index 0. If by some reason your iGPU is listed first under 'Display Adapters' in Device manager, edit the `run-applio-amd.bat` file and change the value from "0" to "1". 

Execute `run-applio-amd.bat` to start Applio.

### 6. Initial Compilation

Any time Zluda emulator meets a previously unseen computational task it compiles the kernel code to support it. During this time there's no output and Applio appears to be frozen. The compilation time takes 15..20 minutes.
