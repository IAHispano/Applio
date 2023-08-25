import os
import torch

import gradio as gr
import librosa
import numpy as np
import logging
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
import traceback
from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from i18n import I18nAuto

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

i18n = I18nAuto()
i18n.print()

config = Config()

weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []
hubert_model = None
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))


def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None: 
            del net_g, n_spk, vc, hubert_model, tgt_sr  
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):  
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = input_audio_path[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        audio = librosa.resample(audio, orig_sr=input_audio_path[0], target_sr=16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("在线demo"):
            gr.Markdown(
                value="""
                RVC 在线demo
                """
            )
            sid = gr.Dropdown(label=i18n("Inferencing voice:"), choices=sorted(names))
            with gr.Column():
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Select Speaker/Singer ID:"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
            sid.change(
                fn=get_vc,
                inputs=[sid],
                outputs=[spk_item],
            )
            gr.Markdown(
                value=i18n("Recommended +12 key for male to female conversion, and -12 key for female to male conversion. If the sound range goes too far and the voice is distorted, you can also adjust it to the appropriate range by yourself.")
            )
            vc_input3 = gr.Audio(label="上传音频（长度小于90秒）")
            vc_transform0 = gr.Number(label=i18n("Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"), value=0)
            f0method0 = gr.Radio(
                label=i18n("Select the pitch extraction algorithm:"),
                choices=["pm", "harvest", "crepe"],
                value="pm",
                interactive=True,
            )
            filter_radius0 = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n("If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."),
                value=3,
                step=1,
                interactive=True,
            )
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=i18n("Feature search dataset file path"),
                    value="",
                    interactive=False,
                    visible=False,
                )
            file_index2 = gr.Dropdown(
                label=i18n("Auto-detect index path and select from the dropdown:"),
                choices=sorted(index_paths),
                interactive=True,
            )
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search feature ratio:"),
                value=0.88,
                interactive=True,
            )
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label=i18n("Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:"),
                value=0,
                step=1,
                interactive=True,
            )
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used:"),
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:"),
                value=0.33,
                step=0.01,
                interactive=True,
            )
            f0_file = gr.File(label=i18n("F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:"))
            but0 = gr.Button(i18n("Convert"), variant="primary")
            vc_output1 = gr.Textbox(label=i18n("Output information:"))
            vc_output2 = gr.Audio(label=i18n("Export audio (click on the three dots in the lower right corner to download)"))
            but0.click(
                vc_single,
                [
                    spk_item,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
            )


app.launch()
