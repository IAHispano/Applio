<div align="center">

<h1>Mangio-RVC-Fork (Retrieval-based-Voice-Conversion) 💻 </h1>
A fork of an easy-to-use SVC framework based on VITS with top1 retrieval 💯. <br><br>
<b> 

> 💓 Please support the original [RVC repository](https://www.bilibili.com/video/BV1pm4y1z7Gm/). Without it, obviously this fork wouldn't have been possible. The Mangio-RVC-Fork aims to essentially enhance the features that the original RVC repo has in my own way. Please note that this fork is NOT STABLE and was forked with the intention of experimentation. Do not use this Fork thinking it is a "better" version of the original repo. Think of it more like another "version" of the original repo. Please note that this doesn't have a google colab. If you want to use google colab, go to the original repository. This fork is intended to be used with paperspace and local machines for now.
</b>

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
+ Local inference with the conv2d 'Half' exception fix. apply the argument --use_gfloat to infer-web.py to use this fix.
+ f0 Inference algorithm overhaul: 🌟
  + Added pyworld dio f0 method.
  + Added torchcrepe crepe f0 method. (Increases pitch accuracy and stability ALOT)
  + Added torchcrepe crepe-tiny model. (Faster on inference, but probably worse quality than normal full crepe)
  + Modifiable crepe_hop_length for the crepe algorithm via the web_gui
+ f0 Crepe Pitch Extraction for training. 🌟 (EXPERIMENTAL) Works on paperspace machines but not local mac/windows machines. Potential memory leak. Watch out.
+ Paperspace integration 🌟
  + Paperspace argument on infer-web.py (--paperspace) that shares a gradio link
  + Make file for paperspace users
+ Tensorboard access via Makefile (make tensorboard)
+ Total epoch slider for the training now limited to 10,000 not just 1000.
+ Added CLI functionality
  + added train-index-cli.py to train the feature index without the GUI
  + added extract-small-model.py to extract the small model without the GUI

## This repository has the following features too:
+ Reduce tone leakage by replacing source feature to training-set feature using top1 retrieval;
+ Easy and fast training, even on relatively poor graphics cards;
+ Training with a small amount of data also obtains relatively good results (>=10min low noise speech recommended);
+ Supporting model fusion to change timbres (using ckpt processing tab->ckpt merge);
+ Easy-to-use Webui interface;
+ Use the UVR5 model to quickly separate vocals and instruments.

## Features planned to be added during the fork's development ▶️
+ Improved GUI (More convenience).
+ Automatic removal of old generations to save space.
+ Potentially a pyin f0 method or a hybrid f0 crepe method.
+ More Optimized training on paperspace machines
+ A feature search ratio booster to emphasize the target timbre. 

# About this fork's crepe training: 
Crepe training is still incredibly instable and there's been report of a memory leak. This will be fixed in the future, however it works quite well on paperspace machines. Please note that crepe training adds a little bit of difference against a harvest trained model. Crepe sounds clearer on some parts, but sounds more robotic on some parts too. Both I would say are equally good to train with, but I still think crepe on INFERENCE is not only quicker, but more pitch stable (especially with vocal layers). Right now, its quite stable to train with a harvest model and infer it with crepe. If you are training with crepe however (f0 feature extraction), please make sure your datasets are as dry as possible to reduce artifacts and unwanted harmonics as I assume the crepe pitch estimation latches on to reverb more.

## If you get CUDA issues with crepe training, or pm and harvest etc.
This is due to the number of processes (n_p) being too high. Make sure to cut the number of threads down. Please lower the value of the "Number of CPU Threads to use" slider on the feature extraction GUI.  

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

# Preparation of other Pre-models ⬇️
## Paperspace Users:
```bash
cd Mangio-RVC-Fork
make base # Do only once after cloning this fork (No need to do it again unless pre-models change on hugging face)
```

## Local Users
RVC requires other pre-models to infer and train.
You need to download them from our [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

Here's a list of Pre-models and other files that RVC needs:
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

#If you are using Windows, you may also need this dictionary, skip if FFmpeg is installed
ffmpeg.exe
```
# Running the Web GUI to Infer & Train 💪

## For paperspace users:
```bash
cd Mangio-RVC-Fork
make run
```
Then click the gradio link it provides.

# Inference & Training with CLI 💪 🔠

## Pre-processing the Dataset without the GUI
```bash
# arg 1 = Dataset Path
# arg 2 = Sample Rate
# arg 3 = Number of Threads
# arg 4 = Export Directory (logs/*YOUR DATASET FOLDER NAME*)
# arg 5 = No parallel: True or False
python trainset_preprocess_pipeline_print.py /INSERTDATASETNAMEHERE 40000 8 logs/mi-test True 
```

## f0 Feature Extraction without the GUI
```bash
# arg 1 = Path of model logs (logs/*YOUR MODEL NAME*)
# arg 2 = Number of threads to use
# arg 3 = f0 method: pm, harvest, dio, crepe
# arg 4 = Crepe Hop Length (Value is used if using crepe method)
python extract_f0_print.py logs/mi-test 4 harvest 128
# arg 1 = device
# arg 2 = n_part
# arg 3 = i_part
# arg 4 = GPU Device number ("0")
# arg 5 = Export Directory logs/*MODEL NAME*
python extract_feature_print.py cpu 1 0 0 logs/mi-test
```

## Training without the GUI

```bash
# Arguments
# -e = Name of model
# -sr = Sample Rate
# -f0 = Model has pitch guidance? 1 for yes. 0 for no.
# -bs = Batch size
# -g = GPU card slot
# -te = Total Epochs
# -se = Save epoch interval
# -pg = Pretrained Generator Model Path
# -pd = Pretrained Discriminator Model Path
# -l = Save only latest model? 1 for yes 0 for no
# -c = Cache data in gpu? 1 for yes 0 for no
python train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 8 -g 0 -te 10000 -se 50 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 0 -c 0
```

## Training the Feature Index without the GUI

```bash
# + Mangio-RVC-Fork Feature. Train the index with the CLI
# arg1 = Model Name (name of the model folder in logs) 
python train-index-cli.py mi-test
```

## Extract Model from checkpoint with the GUI

```bash
# + Mangio-RVC-Fork Feature. Extract Small Model from checkpoint from the CLI.
# The small model refers to the model that can be used for inference
# Arguments:
# arg1 = Path of the model checkpoint (g file path)
# arg2 = Model Save Name
# arg3 = Sample Rate: "32k" "40k" or "48k"
# arg4 = Has Pitch guidance (f0)? Either 1 for yes or 0 for no
# arg5 = Model Information. (OPTIONAL). 
python extract-small-model-cli.py logs/G_99750.pth MyModel 40k 1 "This is a cool model."
```

# Running the Tensorboard 📉
```bash
cd Mangio-RVC-Fork
make tensorboard
```
Then click the tensorboard link it provides and refresh the data.

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

