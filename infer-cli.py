# Fork Feature Mangio RVC Fork. Infer Audio with just the CLI

import torch, os, traceback, sys, warnings, shutil, numpy as np

from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from config import Config
from my_utils import load_audio

# Fork Feature. Write an audio file
from scipy.io.wavfile import write

config = Config(is_gui=False)


weight_root = 'weights'

n_spk = None # Set from get_vc
tgt_sr = 0 # Set from get_vc
net_g = None # Set from get_vc
vc = None # Set from get_vc
cpt = None # Set from get_vc

hubert_model = None # Set from vc_single

def get_hubert():
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
    return hubert_model

def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt
    if sid == "":
        global hubert_model
        if hubert_model != None:
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if_f0 = cpt.get("f0", 1)
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(
                    *cpt["config"], is_half=config.is_half
                )
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 1:
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False)) 
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    print("Mangio-RVC-Fork Infer-CLI: Model has been loaded...")
    return {"visible": True, "maximum": n_spk, "__type__": "update"}

def vc_single(
    sid,
    input_audio,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    crepe_hop_length,
):
    global tgt_sr, net_g, vc, hubert_model, cpt
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            hubert_model = get_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            crepe_hop_length,
            f0_file=f0_file,
        )
        print(
            "npy: ", times[0], "s, f0: ", times[1], "s, infer: ", times[2], "s", sep=""
        )
        return "Success", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def start_inference():
    # Get Essential Paths first
    model_name = str(sys.argv[1]) # MyModel.pth
    source_audio_path = str(sys.argv[2]) # Source Audio Path
    output_file_name = str(sys.argv[3]) # Output audio path e.g outputs/conversion_out.wav
    feature_index_path = str(sys.argv[4]) # Feature Index file path
    f0_file = None # Not implemented yet. To be implemented later on

    # Get parameters for inference
    speaker_id = int(sys.argv[5]) # 0
    transposition = float(sys.argv[6]) # 0.0 float
    f0_method = str(sys.argv[7]) # harvest
    crepe_hop_length = int(sys.argv[8]) # 128
    feature_ratio = float(sys.argv[9]) # 0.78

    # Get VC first. set global vc to VC from pipeline script
    print("Mangio-RVC-Fork Infer-CLI: Starting the inference...")
    vc_data = get_vc(model_name)
    print(vc_data)
    print("Mangio-RVC-Fork Infer-CLI: Performing inference...")
    conversion_data = vc_single(
        speaker_id,
        source_audio_path,
        transposition,
        f0_file,
        f0_method,
        feature_index_path,
        feature_ratio,
        crepe_hop_length
    )
    if(conversion_data[0] == "Success"):
        print("Mangio-RVC-Fork Infer-CLI: Inference succeeded. Writing to %s/%s..." % ('audio-outputs', output_file_name))
        # Go ahead with output
        write('%s/%s' % ('audio-outputs', output_file_name), conversion_data[1][0], conversion_data[1][1])
        print("Mangio-RVC-Fork Infer-CLI: Finished! Saved output to %s/%s" % ('audio-outputs', output_file_name))
    else:
        print("Mangio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ")
        print(conversion_data[0])

start_inference()