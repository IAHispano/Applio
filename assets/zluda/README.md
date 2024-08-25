## Installation and Setup Instructions

### 1. Install VC++ Runtime
Download and install the VC++ Runtime from [this link](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 2. Install HIP SDK

- **Read the [System Requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html)**

  Check the **"Windows-supported GPUs"** section to determine the correct installation steps:

  - **If your GPU has a green checkbox in the HIP SDK column:**
    - **Install either v6.1.2 or v5.7.1 HIP SDK**
      - Download from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
  
  - **If your GPU has a red cross in the HIP SDK column:**
    - **Install v5.7.1 HIP SDK**
      - For 6700, 6700XT, 6750XT, download [this archive](https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1031.7z)
      - For 6600, 6600XT, 6650XT, download [this archive](https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1032.7z)
      
      **Steps:**
      1. Rename `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` to `library.old`
      2. Create a new folder named `library`
      3. Unzip the content of the archive into that folder

  - **If your GPU is not listed:**
    - **Install v5.7.1 HIP SDK**
      1. Google "techpowerup your_gpu" to find the value of "Shader ISA" (gfxnnnn). Only `gfx803/900/906/1010/1011/1012/1030/1100/1101/1102` are supported.
      2. Download [this archive](https://github.com/brknsoul/ROCmLibs/raw/main/ROCmLibs.7z)
      
      **Steps:**
      1. Rename `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library` to `library.old`
      2. Unzip the content of the archive into `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\`

### 3. Patching Applio

1. **Move all `.bat` files from the `zluda` folder to the root directory of Applio.**

2. **For Precompiled Applio:**
   - Run `reinstall-torch.bat` to patch Applio.

3. **For Applio Source Code:**
   1. Open `run-install.bat` and update the Torch versions on line 67:
      ```sh
      pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
      ```
   2. Execute `run-install.bat` to install the required libraries.
   3. Manually apply the code changes from the pull request.

### 4. Download Zluda and Patch Torch Libraries

1. **For HIP SDK 5.7:**
   - Run `patch_zluda_hip57.bat`.
   - Add `C:\Program Files\AMD\ROCm\5.7\bin` to your system's Path environment variable.

2. **For HIP SDK 6.1:**
   - Run `patch_zluda_hip61.bat`.
   - Add `C:\Program Files\AMD\ROCm\6.1\bin` to your system's Path environment variable.

### 5. Starting Applio

- Execute `run-applio-zluda.bat` to start Applio.

