# 8/17 Changelog
- First model + index preloads on boot so you can inference immediately (someguy.mp3 is now the default file too)
- Index search bug where it would always expect eXXX_sXXX pattern is fixed
- Batch inference overhauls in layout and exporting now directly exports as the filetype
- Hertz is the default unit for pitch ranges now + it uses global setting on boot
- Misc translation adjustments to be more concise and minor extra error handling
- Inference prints out progress to the console

# 8/15 Changelog
- Refactor entire inference interface to be much more user friendly
- Make rmvpe+ default (this sets the specific pitch range of 50-1100hz like all other methods do)
- Other misc translation adjustments
- Error handling for when torch hasn't fully loaded + mismatch version index

# 8/6 Changelog
- Completely rewrote code for training related stripts, `train_nsf_sim_cache_sid_load_pretrain.py`, `trainset_preprocess_pipeline_print.py`, etc
- Got rid of CSV database (mostly, still use STOP.csv cause Popen is trash and can't access a global var file when it is changed, for some reason)
- Training is faster now (cool)
- Now auto-indexing involves path like `logs/MODEL/INDEX`, not `./logs/MODEL/INDEX`
- Now you can specify desired Gradio theme without trying to find it in `infer-web.py`! For detailed instructions view `go-web.bat` file.

# 8/1 Changelog
- Completely rewrote the code for `infer-web.py` to be more optimized.
- Added minor optimizations in `my_utils.py`, as well as a new function to gauge if a file is long enough to train with (0.76s)
- Due to optimization, inferencing seems to be decently faster, while training seems to be marginally faster.
- The launch speed is increased due to lazy importing heavy modules and libraries.
- Requirements.txt builds properly on non-windows again + plus fixed a versioning conflict with protobuf

# 7/28 Changelog:
- Merging checkpoints bug has been resolved.
- Revoked from using **SQL** databases back to **CSV** data storage for less trouble on colab.

# 7/26 Changelog:
- Fixed the cli inferencing traceback.
- SQL Databases implemented for formanting training stop.
- Gradio browser tab renamed to `Mangio-RVC-Web 💻`.
- Rudimentary functions from `infer-web.py` removed.
- Formanting now accepts any audio format, as long as it is supported by FFmpeg.

# 7/25 Changelog:
- Better MacOS installation script. For inference, all that needs to be done is running the `run.sh` from the extracted zip folder, where it will install Python 3.8, Homebrew, and other dependencies for you automatically. M1 Macs are natively supported for GPU acceleration, and training should work if you choose to download the pretrained models.

# 7/23 Changelog:
- Fp16 detection now works how it did before the last RVC beta; no more training slowdowns compared to how it was before then
- Logging on TensorBoard is now done per epoch. It logs every X steps in an epoch; therefore each point on the graph is now represented per-epoch.
- Updated UI to include placeholders.
- Certain sliders now appear only when needed.
- Step 2b (feature extract) is hidden if pitch guidance is disabled.
- Unnecessary radios have been replaced with checkboxes.

# 7/22 Changelog:
- Experimental Formant Shift using StftPitchShift (tried using praat with praatio but to no avail)
- Added `Stop Training` button when training, no need to restart RVC every time you want to stop the training of a model!
- Auto-detect Index path for models selected + Auto-detect paths, no more default values like this: `E:\codes\py39\vits_vc_gpu_train\logs\mi-test-1key\total_fea.npy`, We're getting Root Dir and subfolders using 
```python 
os.path.abspath(os.getcwd())
``` 
- Audio files dropdown by autodetecting files in `/audios/` folder 
- More stable version of Gradio(3.34.0) with support of Themes
- Removed `One Click Training` button in `Training` Tab, since it was glitchy and has confused a lot of users. 
- Changed default training settings to be more optimal for newer users. 
- Auto-open TensorBoard localhost URL when `tensor-launch.py` is executed 
- RMVPE implemented in both inferencing and training


<div align="center">
<h1>Mangio-RVC-Fork with v2 Support! 💻 </h1>
A fork of an easy-to-use SVC framework based on VITS with top1 retrieval 💯. In general, this fork provides a CLI interface in addition. And also gives you more f0 methods to use, as well as a personlized 'hybrid' f0 estimation method using nanmedian. <br><br>
<b> 

<h1> !! Feature implementations have been post-poned -- working on other related improvements !! </h1>

<b>Im developing alone. Help support me in the rapid development of open-source A.I Deep NN audio tools and frameworks. By donating, you making a commitment to the improvement of audio A.I as a whole. You should donate if you merely want to "support" me, not because you are then expecting a finished product. There are no guarantees. Thank you</b>
<br>
<a href="https://www.paypal.com/donate/?business=HEW6P8R79NFFN&no_recurring=0&item_name=I+have+an+altruistic+drive+to+develop+open-source+A.I+frameworks+and+tools.+Support+me+in+improving+A.I+audio+entirely.&currency_code=AUD"><img src="./mangio_utils/donate.png" height="42"></a>

> 💓 Please support the original [RVC repository](https://www.bilibili.com/video/BV1pm4y1z7Gm/). Without it, obviously this fork wouldn't have been possible. The Mangio-RVC-Fork aims to essentially enhance the features that the original RVC repo has in my own way. Please note that this fork is NOT STABLE and was forked with the intention of experimentation. Do not use this Fork thinking it is a "better" version of the original repo. Think of it more like another "version" of the original repo. Please note that this doesn't have a google colab. If you want to use google colab, go to the original repository. This fork is intended to be used with paperspace and local machines for now.
</b>

<b> Now supports version 2 pre-trained models! </b>

## Add me on discord: Funky Town#2048
I am able to communicate with you here and there.

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)
  
[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)

[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

Special thanks to discord user @kalomaze#2983 for creating a temporary colab notebook for this fork for the time being. Eventually, an official, more stable notebook will be included with this fork. Please use paperspace instead if you can as it is much more stable.
<br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/drive/1iWOLYE9znqT6XE5Rw2iETE19ZlqpziLx?usp=sharing)

<img src="https://33.media.tumblr.com/2344c05b16d25e60771d315604f90258/tumblr_nmy9cc0Zm71syljkjo1_1280.gif" /><br>
  
</div>

------

> The original RVC [Demo Video](https://www.bilibili.com/video/BV1pm4y1z7Gm/) here!

> Realtime Voice Conversion Software using RVC : [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> The dataset for the pre-training model uses nearly 50 hours of high quality VCTK open source dataset.

> High quality licensed song datasets will be added to training-set one after another for your use, without worrying about copyright infringement.
# Summary 📘
## Features that this fork (Mangio-RVC-Fork) has that the original repo doesn't ☑️
+ f0 Inference algorithm overhaul: 🌟
  + Added pyworld dio f0 method.
  + Added another way of calculating crepe f0 method. (mangio-crepe)
  + Added torchcrepe crepe-tiny model. (Faster on inference, but probably worse quality than normal full crepe)
  + Modifiable crepe_hop_length for the crepe algorithm via the web_gui and CLI. 
+ f0 Crepe Pitch Extraction for training. 🌟 (EXPERIMENTAL) Works on paperspace machines but not local mac/windows machines. Potential memory leak. Watch out.
+ Paperspace integration 🌟
  + Paperspace argument on infer-web.py (--paperspace) that shares a gradio link
  + Make file for paperspace users
+ Tensorboard access via Makefile (make tensorboard)
+ Total epoch slider for the training now limited to 10,000 not just 1000.
+ Added CLI functionality
  + added --is_cli flag on infer-web.py to use the CLI system.
+ f0 hybrid (median) estimation method by calculating nanmedian for a specified array of f0 methods to get the best of all worlds for all specified f0 methods. Only for CLI right now. Soon to be implemented into GUI 🌟
+ f0 hybrid (median) estimation method on f0 feature extraction (training). (VERY EXPERIMENTAL PROBABLY EXTREMELY BUGGY). Feature extraction with the hybrid method will take MUCH longer.

## This repository has the following features too:
+ Reduce tone leakage by replacing source feature to training-set feature using top1 retrieval;
+ Easy and fast training, even on relatively poor graphics cards;
+ Training with a small amount of data also obtains relatively good results (>=10min low noise speech recommended);
+ Supporting model fusion to change timbres (using ckpt processing tab->ckpt merge);
+ Easy-to-use Webui interface;
+ Use the UVR5 model to quickly separate vocals and instruments.

## Features planned to be added during the fork's development ▶️
+ An inference batcher script. Improvement Proposal:
  + According to various github users, apparently inferencing 30 second samples at a time both increases quality and prevents harvest memory errors.
+ Improved GUI (More convenience).
+ Automatic removal of old generations to save space.
+ More Optimized training on paperspace machines 

# About this fork's f0 Hybrid feature on inference
Right now, the hybrid f0 method is only available on CLI, not GUI yet. But basically during inference, we can specify an array of f0 methods, E.G ["pm", "harvest", "crepe"], get f0 calculations of all of them and 'combine' them with nanmedian to get a hybrid f0 signal to get the 'best of all worlds' of the f0 methods provided.

Here's how we would infer with the hybrid f0 method in cli:
```bash
MyModel.pth saudio/Source.wav Output.wav logs/mi-test/added.index 0 -2 hybrid[pm+crepe] 128 3 0 1 0.95 0.33
```
Notice that the method is "hybrid[pm+crepe]" instead of a singular method like "harvest".


```bash
hybrid[pm+harvest+crepe]
# the crepe calculation will be at the 'end' of the computational stack.
# the parselmouth calculation will be at the 'start' of the computational stack.
# the 'hybrid' method will calculate the nanmedian of both pm, harvest and crepe
```

Many f0 methods may be used. But are to be split with a delimiter of the '+' character. Keep in mind that inference will take much longer as we are calculating f0 X more times.

# About the original repo's crepe method, compared to this forks crepe method (mangio-crepe)
The original repos crepe f0 computation method is slightly different to mine. Its arguable that in some areas, my crepe implementation sounds more stable in some parts. However, the orginal repo's crepe implementation gets rid of noise and artifacts much better. In this fork, my own crepe implementation (mangio-crepe) uses a customizable crepe_hop_length feature on both the GUI and the CLI which the original crepe doesnt have. Please let it be known, that each implementation sounds slightly different, and there isn't a clear "better" or "worse". It all depends on the context!

If one must be chosen, I highly recommend using the original crepe implementation (not this fork's) as the developers of RVC have more control on fixing issues than I have.

# About this fork's f0 training additional features.
## Crepe f0 feature extraction
Crepe training is still incredibly instable and there's been report of a memory leak. This will be fixed in the future, however it works quite well on paperspace machines. Please note that crepe training adds a little bit of difference against a harvest trained model. Crepe sounds clearer on some parts, but sounds more robotic on some parts too. Both I would say are equally good to train with, but I still think crepe on INFERENCE is not only quicker, but more pitch stable (especially with vocal layers). Right now, its quite stable to train with a harvest model and infer it with crepe. If you are training with crepe however (f0 feature extraction), please make sure your datasets are as dry as possible to reduce artifacts and unwanted harmonics as I assume the crepe pitch estimation latches on to reverb more.

## Hybrid f0 feature extraction
Only for CLI (not implemented in GUI yet). Basically the same as usage described in this readme's f0 hybrid on inference section. Instead of stating "harvest" into your arguments in the f0 feature extraction page, you would use "hybrid[harvest+dio+pm+crepe]" for example. This f0 nanmedian hybrid method will take very long during feature extraction. Please, if you're willing to use hybrid f0, be patient.

## If you get CUDA issues with crepe training, or pm and harvest etc.
This is due to the number of processes (n_p) being too high. Make sure to cut the number of threads down. Please lower the value of the "Number of CPU Threads to use" slider on the feature extraction GUI.  

# Version Notes
Welcome to RVC version 2!

Please note that version 2 pre-trained models only support 40k model sample rates. If you want to use 32k or 48k however, please use version 1 pre-trained models.

# Installing the Dependencies 🖥️
Using pip (python3.9.8 is stable with this fork)

## Paperspace Users:
```bash
cd Mangio-RVC-Fork
make install # Do this everytime you start your paperspace machine
```

## Windows/MacOS
**Notice**: `faiss 1.7.2` will raise Segmentation Fault: 11 under `MacOS`, please use `pip install faiss-cpu==1.7.0` if you use pip to install it manually.  `Swig` can be installed via `brew` under `MacOS`
 
 ```bash
 brew install swig
 ```

Install requirements:
```bash
pip install -r requirements.txt
```

If you're experiencing httpx invalid port errors please insteall httpx==0.23.0

# Preparation of other Pre-models ⬇️
## Paperspace Users:
```bash
cd Mangio-RVC-Fork
# Do only once after cloning this fork (No need to do it again unless pre-models change on hugging face)
make basev1 
# or if using version 2 pre-trained models.
make basev2
```

## Local Users
RVC requires other pre-models to infer and train.
You need to download them from our [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

Here's a list of Pre-models and other files that RVC needs:
```bash
hubert_base.pt

./pretrained 

./uvr5_weights


If you want to test the v2 version model (the v2 version model changes the feature from the 256-dimensional input of 9-layer hubert+final_proj to the 768-dimensional input of 12-layer hubert, and adds 3 cycle discriminators), an additional download is required

./pretrained_v2 

#If you are using Windows, you may need this file, skip if ffmpeg and ffprobe are installed; ubuntu/debian users can install these two libraries through apt install ffmpeg
./ffmpeg

./ffprobe
```
# Running the Web GUI to Infer & Train 💪

## For paperspace users:
```bash
cd Mangio-RVC-Fork
make run-ui
```
Then click the gradio link it provides.

## Or manually

```bash
# use --paperspace or --colab if on cloud system
python infer-web.py --pycmd python --port 3000
```

# Inference & Training with CLI 💪 🔠
## Paperspace users
```bash
cd Mangio-RVC-Fork
make run-cli
```
## Or Manually
```bash
python infer-web.py --pycmd python --is_cli
```
## Usage
```bash
Mangio-RVC-Fork v2 CLI App!

Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.

You are currently in 'HOME':
    go home            : Takes you back to home with a navigation list.
    go infer           : Takes you to inference command execution.

    go pre-process     : Takes you to training step.1) pre-process command execution.
    go extract-feature : Takes you to training step.2) extract-feature command execution.
    go train           : Takes you to training step.3) being or continue training command execution.
    go train-feature   : Takes you to the train feature index command execution.

    go extract-model   : Takes you to the extract small model command execution.

HOME:
```

Typing 'go infer' for example will take you to the infer page where you can then enter in your arguments that you wish to use for that specific page. For example typing 'go infer' will take you here:

```bash
HOME: go infer
You are currently in 'INFER':
    arg 1) model name with .pth in ./weights: mi-test.pth
    arg 2) source audio path: myFolder\MySource.wav
    arg 3) output file name to be placed in './audio-outputs': MyTest.wav
    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index
    arg 5) speaker id: 0
    arg 6) transposition: 0
    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny)
    arg 8) crepe hop length: 160
    arg 9) harvest median filter radius: 3 (0-7)
    arg 10) post resample rate: 0
    arg 11) mix volume envelope: 1
    arg 12) feature index ratio: 0.78 (0-1)
    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)

Example: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2

INFER: <INSERT ARGUMENTS HERE OR COPY AND PASTE THE EXAMPLE>
```

# Running the Tensorboard 📉
```bash
cd Mangio-RVC-Fork
make tensorboard
```
Then click the tensorboard link it provides and refresh the data.

## Change Gradio Theme:

- [OPTIONAL] Change Gradio's theme:
  1. Go to [Gradio Theme Gallery](https://huggingface.co/spaces/gradio/theme-gallery):
       
  2. Select any theme you like (e.g. [this one](https://huggingface.co/spaces/freddyaboulton/dracula_revamped))
  
  3. Look at the top of the page
  
    ![image](https://github.com/alexlnkp/Mangio-RVC-Tweaks/assets/79400603/59e3e6a9-bdda-4ede-8161-00ee957c1715)

  4. Copy theme variable(in this case, it's `theme='freddyaboulton/dracula_revamped'`)
  
  5. Replace the THEME variable in the `go-web.bat` file with chosen theme. In this case, make it like this: `SET THEME="freddyaboulton/dracula_revamped"`


# Formant Shift Explanation

![image](https://github.com/Mangio621/Mangio-RVC-Fork/assets/79400603/300ebce2-36c7-4761-b1dd-b31403ad2cd1)

- ### Click `Apply` button every time you change the values for inferencing.

- ### As the name implies, you can only use `wav` files so far, also it is very slow, so be patient.

- ### If you added a new `preset.txt` in the `\formantshiftcfg\` folder, click button with refresh emoji

- ### If the preset you selected somehow got edited, by pressing refresh emoji button you'll update values, by grabbing them from the file


# Other 

If you are using Windows, you can download and extract `RVC-beta.7z` to use RVC directly and use `go-web.bat` to start Webui.

There's also a tutorial on RVC in Chinese and you can check it out if needed.

## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
## Thanks to all contributors for their efforts

<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>

