"""Curated UVR model catalog for Applio.

Filenames and URLs below come from the official UVR model indexes
(TRvlvr/model_repo + TRvlvr/application_data). Only public, non-VIP models.
"""

UVR_REPO_PREFIX = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
)
FB_DEMUCS_PREFIX = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"


class UvrModel:
    def __init__(
        self, key, label, arch, files, stems, target, blurb, aliases=(), cls=None
    ):
        self.key = key
        self.label = label
        self.arch = arch  # "vr" | "mdx" | "demucs" | "roformer"
        self.files = files  # list of (filename, url)
        self.stems = stems
        self.target = target
        self.blurb = blurb
        self.aliases = tuple(aliases)
        self.cls = cls  # roformer model class: "mel_band" | "bs"

    def to_dict(self):
        return {
            "key": self.key,
            "label": self.label,
            "arch": self.arch,
            "filename": self.files[0][0],
            "stems": self.stems,
            "target": self.target,
            "blurb": self.blurb,
        }


def _u(fname):
    return f"{UVR_REPO_PREFIX}/{fname}"


MODELS = [
    # NOTE: VR (.pth) entries ship in v1; current release files miss the
    # vendored UVR hash tables (re-uploaded after the tables were made), so a
    # VR run may report the file as unsupported until its band config resolves.
    UvrModel(
        "vr-vocals-hp4",
        "VR Vocals (4_HP)",
        "vr",
        [("4_HP-Vocal-UVR.pth", _u("4_HP-Vocal-UVR.pth"))],
        ["vocals", "instrumental"],
        "vocals",
        "Balanced vocal extraction, good default for singing voice.",
    ),
    UvrModel(
        "vr-vocals-bve",
        "VR Vocals Alt (BVE)",
        "vr",
        [("UVR-BVE-4B_SN-44100-1.pth", _u("UVR-BVE-4B_SN-44100-1.pth"))],
        ["vocals", "instrumental"],
        "vocals",
        "Alternative vocal model, stronger on backing vocals.",
    ),
    UvrModel(
        "vr-karaoke-hp5",
        "VR Karaoke (5_HP)",
        "vr",
        [("5_HP-Karaoke-UVR.pth", _u("5_HP-Karaoke-UVR.pth"))],
        ["instrumental", "vocals"],
        "instrumental",
        "Classic karaoke model: clean instrumental output.",
    ),
    UvrModel(
        "vr-dereverb",
        "VR De-Echo De-Reverb",
        "vr",
        [("UVR-DeEcho-DeReverb.pth", _u("UVR-DeEcho-DeReverb.pth"))],
        ["no reverb", "reverb"],
        "no reverb",
        "Removes echo and reverb; great cleanup before training.",
    ),
    # --- MDX (.onnx, onnx2torch + torch/CUDA) ---
    UvrModel(
        "mdx-main",
        "MDX-NET Main",
        "mdx",
        [("UVR_MDXNET_Main.onnx", _u("UVR_MDXNET_Main.onnx"))],
        ["vocals", "instrumental"],
        "vocals",
        "Flagship MDX vocal model, high detail.",
    ),
    UvrModel(
        "mdx-kim-vocal2",
        "MDX Kim Vocal 2",
        "mdx",
        [("Kim_Vocal_2.onnx", _u("Kim_Vocal_2.onnx"))],
        ["vocals", "instrumental"],
        "vocals",
        "Community favorite for crisp acapellas.",
    ),
    UvrModel(
        "mdx-inst-hq5",
        "MDX Inst HQ 5",
        "mdx",
        [("UVR-MDX-NET-Inst_HQ_5.onnx", _u("UVR-MDX-NET-Inst_HQ_5.onnx"))],
        ["instrumental", "vocals"],
        "instrumental",
        "High-quality instrumental extraction.",
    ),
    UvrModel(
        "mdx-kara2",
        "MDX Karaoke 2",
        "mdx",
        [("UVR_MDXNET_KARA_2.onnx", _u("UVR_MDXNET_KARA_2.onnx"))],
        ["instrumental", "vocals"],
        "instrumental",
        "Karaoke-tuned instrumental model.",
    ),
    UvrModel(
        "mdx-reverb-hq",
        "MDX Reverb HQ",
        "mdx",
        [("Reverb_HQ_By_FoxJoy.onnx", _u("Reverb_HQ_By_FoxJoy.onnx"))],
        ["reverb", "no reverb"],
        "reverb",
        "Isolates or removes reverb; great cleanup before training.",
    ),
    # --- Demucs v4 (htdemucs, 4 stems) ---
    UvrModel(
        "mdxc-instvoc-hq",
        "MDXC InstVoc HQ",
        "mdxc",
        [
            (
                "MDX23C-8KFFT-InstVoc_HQ.ckpt",
                "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt",
            ),
            (
                "model_2_stem_full_band_8k.yaml",
                "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml",
            ),
        ],
        ["Vocals", "Instrumental"],
        None,
        "MDXC MDX23C vocals/instrumental extraction.",
    ),
    # --- Demucs v4 (htdemucs, 4 stems) ---
    UvrModel(
        "demucs-htdemucs",
        "Demucs v4 (4 stems)",
        "demucs",
        [
            ("955717e8-8726e21a.th", f"{FB_DEMUCS_PREFIX}/955717e8-8726e21a.th"),
            ("htdemucs.yaml", _u("htdemucs.yaml")),
        ],
        ["vocals", "drums", "bass", "other"],
        "vocals",
        "Full band split: vocals, drums, bass and other.",
        aliases=("htdemucs",),
    ),
    # --- MSST Roformer family (imported from the mvsepless catalog) ---
    UvrModel(
        "mbr_vocals_kim",
        "Mel-Band Roformer Vocals by Kimberley Jensen",
        "roformer",
        [
            (
                "mbr_vocals_kim.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_kim.ckpt",
            ),
            (
                "mbr_vocals_kim_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_kim_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
        aliases=("roformer-kim-vocals",),
    ),
    UvrModel(
        "mbr_instvoc_duality1_unwa",
        "Mel-Band Roformer InstVoc Duality v1 by Unwa",
        "roformer",
        [
            (
                "mbr_instvoc_duality1_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instvoc_duality1_unwa.ckpt",
            ),
            (
                "mbr_instvoc_duality1_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instvoc_duality1_unwa_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        None,
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instvoc_duality2_unwa",
        "Mel-Band Roformer InstVoc Duality v2 by Unwa",
        "roformer",
        [
            (
                "mbr_instvoc_duality2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instvoc_duality2_unwa.ckpt",
            ),
            (
                "mbr_instvoc_duality2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instvoc_duality2_unwa_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        None,
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft1_unwa",
        "Mel-Band Roformer Kim FT v1 by Unwa",
        "roformer",
        [
            (
                "mbr_kimft1_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft1_unwa.ckpt",
            ),
            (
                "mbr_kimft1_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft1_unwa_config.yaml",
            ),
        ],
        ["Vocals", "other"],
        "Vocals",
        "MSST mel band roformer model (Vocals, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft2_unwa",
        "Mel-Band Roformer Kim FT v2 by Unwa",
        "roformer",
        [
            (
                "mbr_kimft2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2_unwa.ckpt",
            ),
            (
                "mbr_kimft2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft2b_unwa",
        "Mel-Band Roformer Kim FT v2 Bleedless by Unwa",
        "roformer",
        [
            (
                "mbr_kimft2b_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2b_unwa.ckpt",
            ),
            (
                "mbr_kimft2b_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2b_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft3_prev_unwa",
        "Mel-Band Roformer Kim FT v3 preview by Unwa",
        "roformer",
        [
            (
                "mbr_kimft3_prev_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft3_prev_unwa.ckpt",
            ),
            (
                "mbr_kimft3_prev_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft3_prev_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta1_unwa",
        "Mel-Band Roformer Big Beta v1 by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta1_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta1_unwa.ckpt",
            ),
            (
                "mbr_bigbeta1_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta1_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta2_unwa",
        "Mel-Band Roformer Big Beta v2 by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta2_unwa.ckpt",
            ),
            (
                "mbr_bigbeta2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta2_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta3_unwa",
        "Mel-Band Roformer Big Beta v3 by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta3_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta3_unwa.ckpt",
            ),
            (
                "mbr_bigbeta3_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta3_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta4_unwa",
        "Mel-Band Roformer Big Beta v4 by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta4_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta4_unwa.ckpt",
            ),
            (
                "mbr_bigbeta4_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta4_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta5e_unwa",
        "Mel-Band Roformer Vocals Big Beta v5e by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta5e_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta5e_unwa.ckpt",
            ),
            (
                "mbr_bigbeta5e_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta5e_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta6_unwa",
        "Mel-Band Roformer Big Beta v6 by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta6_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta6_unwa.ckpt",
            ),
            (
                "mbr_bigbeta6_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta6_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta6x_unwa",
        "Mel-Band Roformer Big Beta v6x by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta6x_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta6x_unwa.ckpt",
            ),
            (
                "mbr_bigbeta6x_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta6x_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigbeta7_unwa",
        "Mel-Band Roformer Big Beta v7 by Unwa",
        "roformer",
        [
            (
                "mbr_bigbeta7_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta7_unwa.ckpt",
            ),
            (
                "mbr_bigbeta7_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigbeta7_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst1_unwa",
        "Mel-Band Roformer Instrumental v1 by Unwa",
        "roformer",
        [
            (
                "mbr_inst1_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1_unwa.ckpt",
            ),
            (
                "mbr_inst1_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst1+_unwa",
        "Mel-Band Roformer Instrumental v1+ by Unwa",
        "roformer",
        [
            (
                "mbr_inst1+_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1+_unwa.ckpt",
            ),
            (
                "mbr_inst1+_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1+_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst1e_unwa",
        "Mel-Band Roformer Instrumental v1e by Unwa",
        "roformer",
        [
            (
                "mbr_inst1e_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1e_unwa.ckpt",
            ),
            (
                "mbr_inst1e_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1e_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst1e+_unwa",
        "Mel-Band Roformer Instrumental v1e Plus by Unwa",
        "roformer",
        [
            (
                "mbr_inst1e+_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1e+_unwa.ckpt",
            ),
            (
                "mbr_inst1e+_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst1e+_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst2_unwa",
        "Mel-Band Roformer Instrumental v2 by Unwa",
        "roformer",
        [
            (
                "mbr_inst2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst2_unwa.ckpt",
            ),
            (
                "mbr_inst2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst2_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_small_unwa",
        "Mel-Band Roformer Small v1 by Unwa",
        "roformer",
        [
            (
                "mbr_small_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_small_unwa.ckpt",
            ),
            (
                "mbr_small_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_small_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bleed_supressor_unwa_97chris",
        "Mel-Band Roformer Bleed Suppressor v1 by Unwa / 97chris",
        "roformer",
        [
            (
                "mbr_bleed_supressor_unwa_97chris.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bleed_supressor_unwa_97chris.ckpt",
            ),
            (
                "mbr_bleed_supressor_unwa_97chris_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bleed_supressor_unwa_97chris_config.yaml",
            ),
        ],
        ["Bleed", "Instrumental"],
        "Instrumental",
        "MSST mel band roformer model (Bleed, Instrumental).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst_becruily",
        "Mel-Band Roformer Instrumental by Becruily",
        "roformer",
        [
            (
                "mbr_inst_becruily.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_becruily.ckpt",
            ),
            (
                "mbr_inst_becruily_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_becruily_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_guitar_becruily",
        "Mel-Band Roformer Instrumental by Becruily",
        "roformer",
        [
            (
                "mbr_guitar_becruily.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_guitar_becruily.ckpt",
            ),
            (
                "mbr_guitar_becruily_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_guitar_becruily_config.yaml",
            ),
        ],
        ["Guitar", "Other"],
        "Guitar",
        "MSST mel band roformer model (Guitar, Other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_becruily",
        "Mel-Band Roformer Karaoke by Becruily",
        "roformer",
        [
            (
                "mbr_karaoke_becruily.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_becruily.ckpt",
            ),
            (
                "mbr_karaoke_becruily_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_becruily_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        None,
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocals_becruily",
        "Mel-Band Roformer Vocals by Becruily",
        "roformer",
        [
            (
                "mbr_vocals_becruily.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_becruily.ckpt",
            ),
            (
                "mbr_vocals_becruily_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_becruily_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_deux_becruily",
        "Mel-Band Roformer DeUX by Becruily",
        "roformer",
        [
            (
                "mbr_deux_becruily.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_deux_becruily.ckpt",
            ),
            (
                "mbr_deux_becruily_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_deux_becruily_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        None,
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhft1",
        "Mel-Band Roformer SYHFT v1 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhft1.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft1.ckpt",
            ),
            (
                "mbr_syhft1_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft1_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhft2",
        "Mel-Band Roformer SYHFT v2 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhft2.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft2.ckpt",
            ),
            (
                "mbr_syhft2_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft2_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhft2.5",
        "Mel-Band Roformer SYHFT v2.5 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhft2.5.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft2.5.ckpt",
            ),
            (
                "mbr_syhft2.5_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft2.5_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhft3",
        "Mel-Band Roformer SYHFT v3 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhft3.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft3.ckpt",
            ),
            (
                "mbr_syhft3_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft3_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bigsyhft1fast",
        "Mel-Band Roformer Big SYHFT v1 Fast by SYH99999",
        "roformer",
        [
            (
                "mbr_bigsyhft1fast.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigsyhft1fast.ckpt",
            ),
            (
                "mbr_bigsyhft1fast_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bigsyhft1fast_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhftbeta1",
        "Mel-Band Roformer Merged Beta v1 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhftbeta1.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftbeta1.ckpt",
            ),
            (
                "mbr_syhftbeta1_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftbeta1_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhftB1_1",
        "Mel-Band Roformer SYHFT B1 1 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhftB1_1.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftB1_1.ckpt",
            ),
            (
                "mbr_syhftB1_1_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftB1_1_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhftB1_2",
        "Mel-Band Roformer SYHFT B1 2 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhftB1_2.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftB1_2.ckpt",
            ),
            (
                "mbr_syhftB1_2_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftB1_2_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhftB1_3",
        "Mel-Band Roformer SYHFT B1 3 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhftB1_3.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftB1_3.ckpt",
            ),
            (
                "mbr_syhftB1_3_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhftB1_3_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhft_4stem",
        "Mel-Band Roformer 4 Stems FT Large v1 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhft_4stem.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft_4stem.ckpt",
            ),
            (
                "mbr_syhft_4stem_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft_4stem_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST mel band roformer model (bass, drums, other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_syhft_4stem2",
        "Mel-Band Roformer 4 Stems FT Large v2 by SYH99999",
        "roformer",
        [
            (
                "mbr_syhft_4stem2.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft_4stem2.ckpt",
            ),
            (
                "mbr_syhft_4stem2_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_syhft_4stem2_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST mel band roformer model (bass, drums, other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst_1652_essid",
        "Mel-Band Roformer Instrumental by Essid (sdr 16.52)",
        "roformer",
        [
            (
                "mbr_inst_1652_essid.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_1652_essid.ckpt",
            ),
            (
                "mbr_inst_1652_essid_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_1652_essid_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst_1681_essid",
        "Mel-Band Roformer Instrumental by Essid (sdr 16.81)",
        "roformer",
        [
            (
                "mbr_inst_1681_essid.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_1681_essid.ckpt",
            ),
            (
                "mbr_inst_1681_essid_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_1681_essid_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv1_gabox",
        "Mel-Band Roformer Instrumental Fv1 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv1_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv1_gabox.ckpt",
            ),
            (
                "mbr_instfv1_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv1_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv2_gabox",
        "Mel-Band Roformer Instrumental Fv2 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv2_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv2_gabox.ckpt",
            ),
            (
                "mbr_instfv2_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv2_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv3_gabox",
        "Mel-Band Roformer Instrumental Fv3 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv3_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv3_gabox.ckpt",
            ),
            (
                "mbr_instfv3_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv3_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv4_gabox",
        "Mel-Band Roformer Instrumental Fv4 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv4_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv4_gabox.ckpt",
            ),
            (
                "mbr_instfv4_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv4_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv4n_gabox",
        "Mel-Band Roformer Instrumental Fv4 Noise by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv4n_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv4n_gabox.ckpt",
            ),
            (
                "mbr_instfv4n_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv4n_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv5_gabox",
        "Mel-Band Roformer Instrumental Fv5 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv5_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv5_gabox.ckpt",
            ),
            (
                "mbr_instfv5_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv5_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv5n_gabox",
        "Mel-Band Roformer Instrumental Fv5 Noise by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv5n_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv5n_gabox.ckpt",
            ),
            (
                "mbr_instfv5n_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv5n_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv6_gabox",
        "Mel-Band Roformer Instrumental Fv6 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv6_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv6_gabox.ckpt",
            ),
            (
                "mbr_instfv6_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv6_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv6n_gabox",
        "Mel-Band Roformer Instrumental Fv6 Noise by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv6n_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv6n_gabox.ckpt",
            ),
            (
                "mbr_instfv6n_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv6n_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv7_gabox",
        "Mel-Band Roformer Instrumental Fv7 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv7_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7_gabox.ckpt",
            ),
            (
                "mbr_instfv7_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv7n_gabox",
        "Mel-Band Roformer Instrumental Fv7 Noise by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv7n_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7n_gabox.ckpt",
            ),
            (
                "mbr_instfv7n_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7n_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv7+_gabox",
        "Mel-Band Roformer Instrumental Fv7+ by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv7+_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7+_gabox.ckpt",
            ),
            (
                "mbr_instfv7+_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7+_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv7z_gabox",
        "Mel-Band Roformer Instrumental Fv7z by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv7z_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7z_gabox.ckpt",
            ),
            (
                "mbr_instfv7z_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv7z_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv8_gabox",
        "Mel-Band Roformer Instrumental Fv8 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv8_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv8_gabox.ckpt",
            ),
            (
                "mbr_instfv8_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv8_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv8b_gabox",
        "Mel-Band Roformer Instrumental Fv8b by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv8b_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv8b_gabox.ckpt",
            ),
            (
                "mbr_instfv8b_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv8b_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv9_gabox",
        "Mel-Band Roformer Instrumental Fv9 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv9_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv9_gabox.ckpt",
            ),
            (
                "mbr_instfv9_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv9_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv9_2_gabox",
        "Mel-Band Roformer Instrumental Fv9 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv9_2_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv9_2_gabox.ckpt",
            ),
            (
                "mbr_instfv9_2_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv9_2_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfv10_gabox",
        "Mel-Band Roformer Instrumental Fv10 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfv10_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv10_gabox.ckpt",
            ),
            (
                "mbr_instfv10_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfv10_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instflowersv10_gabox",
        "Mel-Band Roformer Instrumental Flowers v10 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instflowersv10_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instflowersv10_gabox.ckpt",
            ),
            (
                "mbr_instflowersv10_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instflowersv10_gabox_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instfvx_gabox",
        "Mel-Band Roformer Instrumental FvX by GaboxR67",
        "roformer",
        [
            (
                "mbr_instfvx_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfvx_gabox.ckpt",
            ),
            (
                "mbr_instfvx_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instfvx_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instbv1_gabox",
        "Mel-Band Roformer Instrumental Bv1 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instbv1_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instbv1_gabox.ckpt",
            ),
            (
                "mbr_instbv1_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instbv1_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instbv2_gabox",
        "Mel-Band Roformer Instrumental Bv2 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instbv2_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instbv2_gabox.ckpt",
            ),
            (
                "mbr_instbv2_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instbv2_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_instbv3_gabox",
        "Mel-Band Roformer Instrumental Bv3 by GaboxR67",
        "roformer",
        [
            (
                "mbr_instbv3_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instbv3_gabox.ckpt",
            ),
            (
                "mbr_instbv3_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_instbv3_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv1_gabox",
        "Mel-Band Roformer Vocals Fv1 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv1_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv1_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv1_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv1_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv2_gabox",
        "Mel-Band Roformer Vocals Fv2 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv2_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv2_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv2_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv2_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv3_gabox",
        "Mel-Band Roformer Vocals Fv3 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv3_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv3_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv3_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv3_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv4_gabox",
        "Mel-Band Roformer Vocals Fv4 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv4_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv4_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv4_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv4_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv5_gabox",
        "Mel-Band Roformer Vocals Fv5 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv5_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv5_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv5_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv5_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv6_gabox",
        "Mel-Band Roformer Vocals Fv6 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv6_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv6_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv6_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv6_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv7_gabox",
        "Mel-Band Roformer Vocals Fv7 Final by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv7_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv7_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv7_beta1_gabox",
        "Mel-Band Roformer Vocals Fv7 Beta 1 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv7_beta1_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_beta1_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv7_beta1_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_beta1_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv7_beta2_gabox",
        "Mel-Band Roformer Vocals Fv7 Beta 2 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv7_beta2_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_beta2_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv7_beta2_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_beta2_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsfv7_beta3_gabox",
        "Mel-Band Roformer Vocals Fv7 Beta 3 by GaboxR67",
        "roformer",
        [
            (
                "mbr_vocalsfv7_beta3_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_beta3_gabox.ckpt",
            ),
            (
                "mbr_vocalsfv7_beta3_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsfv7_beta3_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke25022025_gabox",
        "Mel-Band Roformer Karaoke 25-02-2025 by GaboxR67",
        "roformer",
        [
            (
                "mbr_karaoke25022025_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke25022025_gabox.ckpt",
            ),
            (
                "mbr_karaoke25022025_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke25022025_gabox_config.yaml",
            ),
        ],
        ["karaoke", "other"],
        "karaoke",
        "MSST mel band roformer model (karaoke, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke28022025_gabox",
        "Mel-Band Roformer Karaoke 28-02-2025 by GaboxR67",
        "roformer",
        [
            (
                "mbr_karaoke28022025_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke28022025_gabox.ckpt",
            ),
            (
                "mbr_karaoke28022025_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke28022025_gabox_config.yaml",
            ),
        ],
        ["karaoke", "other"],
        "karaoke",
        "MSST mel band roformer model (karaoke, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke1_gabox",
        "Mel-Band Roformer Karaoke v1 by GaboxR67",
        "roformer",
        [
            (
                "mbr_karaoke1_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke1_gabox.ckpt",
            ),
            (
                "mbr_karaoke1_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke1_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke2_gabox",
        "Mel-Band Roformer Karaoke v2 by GaboxR67",
        "roformer",
        [
            (
                "mbr_karaoke2_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke2_gabox.ckpt",
            ),
            (
                "mbr_karaoke2_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke2_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_small_gabox_aufr33",
        "Mel-Band Roformer Karaoke Small by GaboxR67 & Aufr33",
        "roformer",
        [
            (
                "mbr_karaoke_small_gabox_aufr33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_small_gabox_aufr33.ckpt",
            ),
            (
                "mbr_karaoke_small_gabox_aufr33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_small_gabox_aufr33_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_leadvoc_dereverb_gabox",
        "Mel-Band Roformer Lead Vocals DeReverb by GaboxR67",
        "roformer",
        [
            (
                "mbr_leadvoc_dereverb_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_leadvoc_dereverb_gabox.ckpt",
            ),
            (
                "mbr_leadvoc_dereverb_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_leadvoc_dereverb_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_denoise_debleed_gabox",
        "Mel-Band Roformer Denoise DeBleed by GaboxR67",
        "roformer",
        [
            (
                "mbr_denoise_debleed_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_debleed_gabox.ckpt",
            ),
            (
                "mbr_denoise_debleed_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_debleed_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_fusion_gonzaluigi",
        "Mel-Band Roformer Karaoke Fusion by Gonzaluigi",
        "roformer",
        [
            (
                "mbr_karaoke_fusion_gonzaluigi.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion_gonzaluigi.ckpt",
            ),
            (
                "mbr_karaoke_fusion_gonzaluigi_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion_gonzaluigi_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_fusion_aggr_gonzaluigi",
        "Mel-Band Roformer Karaoke Fusion Aggressive by Gonzaluigi",
        "roformer",
        [
            (
                "mbr_karaoke_fusion_aggr_gonzaluigi.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion_aggr_gonzaluigi.ckpt",
            ),
            (
                "mbr_karaoke_fusion_aggr_gonzaluigi_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion_aggr_gonzaluigi_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bve_gonzaluigi",
        "Mel-Band Roformer BVE by Gonzaluigi",
        "roformer",
        [
            (
                "mbr_bve_gonzaluigi.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bve_gonzaluigi.ckpt",
            ),
            (
                "mbr_bve_gonzaluigi_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bve_gonzaluigi_config.yaml",
            ),
        ],
        ["Back", "Lead"],
        "Lead",
        "MSST mel band roformer model (Back, Lead).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_fusion2_aggr_gonzaluigi",
        "Mel-Band Roformer Karaoke Fusion Aggressive by Gonzaluigi",
        "roformer",
        [
            (
                "mbr_karaoke_fusion2_aggr_gonzaluigi.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion2_aggr_gonzaluigi.ckpt",
            ),
            (
                "mbr_karaoke_fusion2_aggr_gonzaluigi_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion2_aggr_gonzaluigi_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_fusion_total_aggr_gonzaluigi",
        "Mel-Band Roformer Karaoke Fusion Total by Gonzaluigi",
        "roformer",
        [
            (
                "mbr_karaoke_fusion_total_aggr_gonzaluigi.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion_total_aggr_gonzaluigi.ckpt",
            ),
            (
                "mbr_karaoke_fusion_total_aggr_gonzaluigi_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_fusion_total_aggr_gonzaluigi_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_dereverb_anvuew",
        "Mel-Band Roformer DeReverb by Anvuew",
        "roformer",
        [
            (
                "mbr_dereverb_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_anvuew.ckpt",
            ),
            (
                "mbr_dereverb_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_anvuew_config.yaml",
            ),
        ],
        ["noreverb", "reverb"],
        "noreverb",
        "MSST mel band roformer model (noreverb, reverb).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_dereverb_less_aggr_anvuew",
        "Mel-Band Roformer DeReverb Less Aggressive by Anvuew",
        "roformer",
        [
            (
                "mbr_dereverb_less_aggr_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_less_aggr_anvuew.ckpt",
            ),
            (
                "mbr_dereverb_less_aggr_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_less_aggr_anvuew_config.yaml",
            ),
        ],
        ["noreverb", "reverb"],
        "noreverb",
        "MSST mel band roformer model (noreverb, reverb).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_dereverb_mono_anvuew",
        "Mel-Band Roformer DeReverb Mono by Anvuew",
        "roformer",
        [
            (
                "mbr_dereverb_mono_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_mono_anvuew.ckpt",
            ),
            (
                "mbr_dereverb_mono_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_mono_anvuew_config.yaml",
            ),
        ],
        ["noreverb", "reverb"],
        "noreverb",
        "MSST mel band roformer model (noreverb, reverb).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_aspiration_sucial",
        "Mel-Band Roformer Aspiration by Sucial",
        "roformer",
        [
            (
                "mbr_aspiration_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_aspiration_sucial.ckpt",
            ),
            (
                "mbr_aspiration_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_aspiration_sucial_config.yaml",
            ),
        ],
        ["aspiration", "other"],
        None,
        "MSST mel band roformer model (aspiration, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_dereverb_echo1_sucial",
        "Mel-Band Roformer DeReverb-Echo by Sucial",
        "roformer",
        [
            (
                "mbr_dereverb_echo1_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_echo1_sucial.ckpt",
            ),
            (
                "mbr_dereverb_echo1_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb_echo1_sucial_config.yaml",
            ),
        ],
        ["dry", "other"],
        None,
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_debigreverb_sucial",
        "Mel-Band Roformer DeBigReverb by Sucial",
        "roformer",
        [
            (
                "mbr_debigreverb_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_debigreverb_sucial.ckpt",
            ),
            (
                "mbr_debigreverb_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_debigreverb_sucial_config.yaml",
            ),
        ],
        ["dry", "other"],
        "dry",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_desuperbigreverb_sucial",
        "Mel-Band Roformer Super Big DeReverb by Sucial",
        "roformer",
        [
            (
                "mbr_desuperbigreverb_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_desuperbigreverb_sucial.ckpt",
            ),
            (
                "mbr_desuperbigreverb_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_desuperbigreverb_sucial_config.yaml",
            ),
        ],
        ["dry", "other"],
        "dry",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_dereverb-echo_fused_sucial",
        "Mel-Band Roformer DeReverb-Echo Fused by Sucial",
        "roformer",
        [
            (
                "mbr_dereverb-echo_fused_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb-echo_fused_sucial.ckpt",
            ),
            (
                "mbr_dereverb-echo_fused_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb-echo_fused_sucial_config.yaml",
            ),
        ],
        ["dry", "other"],
        "dry",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_dereverb-echo2_sucial",
        "Mel-Band Roformer DeReverb-Echo v2 by Sucial",
        "roformer",
        [
            (
                "mbr_dereverb-echo2_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb-echo2_sucial.ckpt",
            ),
            (
                "mbr_dereverb-echo2_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_dereverb-echo2_sucial_config.yaml",
            ),
        ],
        ["dry", "other"],
        "dry",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_karaoke_aufr33_viperx",
        "Mel-Band Roformer Karaoke by Aufr33 & ViperX",
        "roformer",
        [
            (
                "mbr_karaoke_aufr33_viperx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_aufr33_viperx.ckpt",
            ),
            (
                "mbr_karaoke_aufr33_viperx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_karaoke_aufr33_viperx_config.yaml",
            ),
        ],
        ["karaoke", "other"],
        "karaoke",
        "MSST mel band roformer model (karaoke, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_denoise_aufr33",
        "Mel-Band Roformer DeNoise by Aufr33",
        "roformer",
        [
            (
                "mbr_denoise_aufr33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_aufr33.ckpt",
            ),
            (
                "mbr_denoise_aufr33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_aufr33_config.yaml",
            ),
        ],
        ["dry", "other"],
        "dry",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_denoise_aggr_aufr33",
        "Mel-Band Roformer DeNoise Aggressive by Aufr33",
        "roformer",
        [
            (
                "mbr_denoise_aggr_aufr33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_aggr_aufr33.ckpt",
            ),
            (
                "mbr_denoise_aggr_aufr33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_aggr_aufr33_config.yaml",
            ),
        ],
        ["dry", "other"],
        "dry",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_denoise_yuluoye",
        "Mel-Band Roformer DeNoise by Yuluoye",
        "roformer",
        [
            (
                "mbr_denoise_yuluoye.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_yuluoye.ckpt",
            ),
            (
                "mbr_denoise_yuluoye_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_yuluoye_config.yaml",
            ),
        ],
        ["dry", "other"],
        "other",
        "MSST mel band roformer model (dry, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_denoise_children_phaedrus33",
        "Mel-Band Roformer DeNoiser Children 16k by Phaedrus33",
        "roformer",
        [
            (
                "mbr_denoise_children_phaedrus33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_children_phaedrus33.ckpt",
            ),
            (
                "mbr_denoise_children_phaedrus33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_denoise_children_phaedrus33_config.yaml",
            ),
        ],
        ["noise", "speech"],
        "speech",
        "MSST mel band roformer model (noise, speech).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_crowd_aufr33_viperx",
        "Mel-Band Roformer Crowd by Aufr33 & ViperX",
        "roformer",
        [
            (
                "mbr_crowd_aufr33_viperx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_crowd_aufr33_viperx.ckpt",
            ),
            (
                "mbr_crowd_aufr33_viperx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_crowd_aufr33_viperx_config.yaml",
            ),
        ],
        ["crowd", "other"],
        "crowd",
        "MSST mel band roformer model (crowd, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocals_viperx",
        "Mel-Band Roformer Vocals by ViperX",
        "roformer",
        [
            (
                "mbr_vocals_viperx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_viperx.ckpt",
            ),
            (
                "mbr_vocals_viperx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_viperx_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocalsf_aname",
        "Mel-Band Roformer Vocals Fullness by Aname",
        "roformer",
        [
            (
                "mbr_vocalsf_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsf_aname.ckpt",
            ),
            (
                "mbr_vocalsf_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocalsf_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft1_aname",
        "Mel-Band Roformer Kim FT v1 by Aname",
        "roformer",
        [
            (
                "mbr_kimft1_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft1_aname.ckpt",
            ),
            (
                "mbr_kimft1_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft1_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft2_aname",
        "Mel-Band Roformer Kim FT v2 by Aname",
        "roformer",
        [
            (
                "mbr_kimft2_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2_aname.ckpt",
            ),
            (
                "mbr_kimft2_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft2f_aname",
        "Mel-Band Roformer Kim FT v2 Fullness by Aname",
        "roformer",
        [
            (
                "mbr_kimft2f_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2f_aname.ckpt",
            ),
            (
                "mbr_kimft2f_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft2f_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_kimft3_aname",
        "Mel-Band Roformer Kim FT v3 by Aname",
        "roformer",
        [
            (
                "mbr_kimft3_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft3_aname.ckpt",
            ),
            (
                "mbr_kimft3_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_kimft3_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_small_aname",
        "Mel-Band Roformer Small by Aname",
        "roformer",
        [
            (
                "mbr_small_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_small_aname.ckpt",
            ),
            (
                "mbr_small_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_small_aname_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        None,
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_duality1_aname",
        "Mel-Band Roformer Duality v1 by Aname",
        "roformer",
        [
            (
                "mbr_duality1_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_duality1_aname.ckpt",
            ),
            (
                "mbr_duality1_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_duality1_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_4stemlarge1_aname",
        "Mel-Band Roformer 4 Stems Large by Aname",
        "roformer",
        [
            (
                "mbr_4stemlarge1_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_4stemlarge1_aname.ckpt",
            ),
            (
                "mbr_4stemlarge1_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_4stemlarge1_aname_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST mel band roformer model (bass, drums, other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_4stemlarge2_aname",
        "Mel-Band Roformer 4 Stems v2 Large by Aname",
        "roformer",
        [
            (
                "mbr_4stemlarge2_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_4stemlarge2_aname.ckpt",
            ),
            (
                "mbr_4stemlarge2_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_4stemlarge2_aname_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST mel band roformer model (bass, drums, other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_4stemxl1_aname",
        "Mel-Band Roformer 4 Stems XL by Aname",
        "roformer",
        [
            (
                "mbr_4stemxl1_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_4stemxl1_aname.ckpt",
            ),
            (
                "mbr_4stemxl1_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_4stemxl1_aname_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST mel band roformer model (bass, drums, other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_hybrid_arch_aname",
        "Mel-Band Roformer Hybrid Arch by Aname",
        "roformer",
        [
            (
                "mbr_hybrid_arch_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_hybrid_arch_aname.ckpt",
            ),
            (
                "mbr_hybrid_arch_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_hybrid_arch_aname_config.yaml",
            ),
        ],
        ["vocals", "other"],
        None,
        "MSST mel band roformer model (vocals, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_scratch_aname",
        "Mel-Band Roformer Scratch Large by Aname",
        "roformer",
        [
            (
                "mbr_scratch_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_scratch_aname.ckpt",
            ),
            (
                "mbr_scratch_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_scratch_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_bgm_jasper",
        "Mel-Band Roformer BGM by Jasper",
        "roformer",
        [
            (
                "mbr_bgm_jasper.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bgm_jasper.ckpt",
            ),
            (
                "mbr_bgm_jasper_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_bgm_jasper_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_percussion_yolkispaliks",
        "Mel-Band Roformer Percussion Experimental by yolkispalkis",
        "roformer",
        [
            (
                "mbr_percussion_yolkispaliks.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_percussion_yolkispaliks.ckpt",
            ),
            (
                "mbr_percussion_yolkispaliks_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_percussion_yolkispaliks_config.yaml",
            ),
        ],
        ["other", "percussions"],
        "percussions",
        "MSST mel band roformer model (other, percussions).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst_metal_prev_meskvlla33",
        "Mel-Band Roformer Metal Inst Preview by Mesk",
        "roformer",
        [
            (
                "mbr_inst_metal_prev_meskvlla33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_metal_prev_meskvlla33.ckpt",
            ),
            (
                "mbr_inst_metal_prev_meskvlla33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_metal_prev_meskvlla33_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_inst_rifforge_meskvlla33",
        "Mel-Band Roformer Inst Rifforge by Mesk",
        "roformer",
        [
            (
                "mbr_inst_rifforge_meskvlla33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_rifforge_meskvlla33.ckpt",
            ),
            (
                "mbr_inst_rifforge_meskvlla33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_inst_rifforge_meskvlla33_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_neo_inst_vfx",
        "Mel-Band Roformer NEO Inst VFX by natanworkspace",
        "roformer",
        [
            (
                "mbr_neo_inst_vfx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_neo_inst_vfx.ckpt",
            ),
            (
                "mbr_neo_inst_vfx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_neo_inst_vfx_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Instrumental",
        "MSST mel band roformer model (Instrumental, Vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_lead_rhythm_guitar_listra92",
        "Mel-Band Roformer Lead-Rhythm Guitar by listra92",
        "roformer",
        [
            (
                "mbr_lead_rhythm_guitar_listra92.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_lead_rhythm_guitar_listra92.ckpt",
            ),
            (
                "mbr_lead_rhythm_guitar_listra92_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_lead_rhythm_guitar_listra92_config.yaml",
            ),
        ],
        ["Lead", "Rhythm"],
        "Lead",
        "MSST mel band roformer model (Lead, Rhythm).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_guitar_chencfd",
        "Mel-Band Roformer Guitar by chenCFD",
        "roformer",
        [
            (
                "mbr_guitar_chencfd.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_guitar_chencfd.ckpt",
            ),
            (
                "mbr_guitar_chencfd_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_guitar_chencfd_config.yaml",
            ),
        ],
        ["guitar", "others"],
        "guitar",
        "MSST mel band roformer model (guitar, others).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_mid_side_gilliaaan",
        "Mel-Band Roformer Mid-Side by Gilliaaan",
        "roformer",
        [
            (
                "mbr_mid_side_gilliaaan.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_mid_side_gilliaaan.ckpt",
            ),
            (
                "mbr_mid_side_gilliaaan_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_mid_side_gilliaaan_config.yaml",
            ),
        ],
        ["mid", "side"],
        None,
        "MSST mel band roformer model (mid, side).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_vocals_zfturbo",
        "Mel-Band Roformer Vocals by ZFTurbo",
        "roformer",
        [
            (
                "mbr_vocals_zfturbo.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_zfturbo.ckpt",
            ),
            (
                "mbr_vocals_zfturbo_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_vocals_zfturbo_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST mel band roformer model (other, vocals).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_amb_jazzpear",
        "Mel-Band Roformer Ambiance by jazzpear",
        "roformer",
        [
            (
                "mbr_amb_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_amb_jazzpear.ckpt",
            ),
            (
                "mbr_amb_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_amb_jazzpear_config.yaml",
            ),
        ],
        ["ambience", "other"],
        "ambience",
        "MSST mel band roformer model (ambience, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_expl_jazzpear",
        "Mel-Band Roformer Explosions by jazzpear",
        "roformer",
        [
            (
                "mbr_expl_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_expl_jazzpear.ckpt",
            ),
            (
                "mbr_expl_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_expl_jazzpear_config.yaml",
            ),
        ],
        ["explosions", "other"],
        "explosions",
        "MSST mel band roformer model (explosions, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_fight_jazzpear",
        "Mel-Band Roformer Fighting by jazzpear",
        "roformer",
        [
            (
                "mbr_fight_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_fight_jazzpear.ckpt",
            ),
            (
                "mbr_fight_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_fight_jazzpear_config.yaml",
            ),
        ],
        ["fighting", "other"],
        "fighting",
        "MSST mel band roformer model (fighting, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_foot_jazzpear",
        "Mel-Band Roformer Footsteps by jazzpear",
        "roformer",
        [
            (
                "mbr_foot_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_foot_jazzpear.ckpt",
            ),
            (
                "mbr_foot_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_foot_jazzpear_config.yaml",
            ),
        ],
        ["footsteps", "other"],
        "footsteps",
        "MSST mel band roformer model (footsteps, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_gen_jazzpear",
        "Mel-Band Roformer General by jazzpear",
        "roformer",
        [
            (
                "mbr_gen_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_gen_jazzpear.ckpt",
            ),
            (
                "mbr_gen_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_gen_jazzpear_config.yaml",
            ),
        ],
        ["vocals", "other"],
        "vocals",
        "MSST mel band roformer model (vocals, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_misc_jazzpear",
        "Mel-Band Roformer Foley by jazzpear",
        "roformer",
        [
            (
                "mbr_misc_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_misc_jazzpear.ckpt",
            ),
            (
                "mbr_misc_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_misc_jazzpear_config.yaml",
            ),
        ],
        ["foley", "other"],
        "foley",
        "MSST mel band roformer model (foley, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_toon_jazzpear",
        "Mel-Band Roformer Toon by jazzpear",
        "roformer",
        [
            (
                "mbr_toon_jazzpear.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_toon_jazzpear.ckpt",
            ),
            (
                "mbr_toon_jazzpear_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_toon_jazzpear_config.yaml",
            ),
        ],
        ["anime", "other"],
        "anime",
        "MSST mel band roformer model (anime, other).",
        cls="mel_band",
    ),
    UvrModel(
        "mbr_speech_alicen",
        "Mel-Band Roformer SpeechSep by AliceN",
        "roformer",
        [
            (
                "mbr_speech_alicen.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_speech_alicen.ckpt",
            ),
            (
                "mbr_speech_alicen_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_speech_alicen_config.yaml",
            ),
        ],
        ["vocals", "other"],
        None,
        "MSST mel band roformer model (vocals, other).",
        cls="mel_band",
    ),
    UvrModel(
        "bs_drums_beatloo_labs",
        "BS Roformer Drums Experimental by BeatLoo Labs",
        "roformer",
        [
            (
                "bs_drums_beatloo_labs.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_beatloo_labs.ckpt",
            ),
            (
                "bs_drums_beatloo_labs_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_beatloo_labs_config.yaml",
            ),
        ],
        ["drums", "other"],
        "drums",
        "MSST bs roformer model (drums, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_bass_beatloo_labs",
        "BS Roformer Bass Experimental by BeatLoo Labs",
        "roformer",
        [
            (
                "bs_bass_beatloo_labs.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_bass_beatloo_labs.ckpt",
            ),
            (
                "bs_bass_beatloo_labs_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_bass_beatloo_labs_config.yaml",
            ),
        ],
        ["bass", "other"],
        "bass",
        "MSST bs roformer model (bass, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocals_1296_viperx",
        "BS Roformer Vocals (sdr 12.96) by ViperX",
        "roformer",
        [
            (
                "bs_vocals_1296_viperx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_1296_viperx.ckpt",
            ),
            (
                "bs_vocals_1296_viperx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_1296_viperx_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST bs roformer model (Instrumental, Vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocals_1297_viperx",
        "BS Roformer Vocals (sdr 12.97) by ViperX",
        "roformer",
        [
            (
                "bs_vocals_1297_viperx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_1297_viperx.ckpt",
            ),
            (
                "bs_vocals_1297_viperx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_1297_viperx_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST bs roformer model (Instrumental, Vocals).",
        cls="bs",
        aliases=("roformer-viperx-1297",),
    ),
    UvrModel(
        "bs_other_viperx",
        "BS Roformer Other by ViperX",
        "roformer",
        [
            (
                "bs_other_viperx.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_other_viperx.ckpt",
            ),
            (
                "bs_other_viperx_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_other_viperx_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_inst_exp_vlp_unwa",
        "BS Roformer Instrumental EXP Value Residual by Unwa",
        "roformer",
        [
            (
                "bs_inst_exp_vlp_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_exp_vlp_unwa.ckpt",
            ),
            (
                "bs_inst_exp_vlp_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_exp_vlp_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_revive1_unwa",
        "BS Roformer Vocals Revive v1 by Unwa",
        "roformer",
        [
            (
                "bs_revive1_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_revive1_unwa.ckpt",
            ),
            (
                "bs_revive1_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_revive1_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_revive2_unwa",
        "BS Roformer Vocals Revive v2 by Unwa",
        "roformer",
        [
            (
                "bs_revive2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_revive2_unwa.ckpt",
            ),
            (
                "bs_revive2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_revive2_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_revive3e_unwa",
        "BS Roformer Vocals Revive v3e by Unwa",
        "roformer",
        [
            (
                "bs_revive3e_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_revive3e_unwa.ckpt",
            ),
            (
                "bs_revive3e_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_revive3e_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocals_large1_unwa",
        "BS Roformer Vocals Large v1 by Unwa",
        "roformer",
        [
            (
                "bs_vocals_large1_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_large1_unwa.ckpt",
            ),
            (
                "bs_vocals_large1_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_large1_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_resurrection_unwa",
        "BS Roformer Vocals Resurrection by Unwa",
        "roformer",
        [
            (
                "bs_resurrection_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_resurrection_unwa.ckpt",
            ),
            (
                "bs_resurrection_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_resurrection_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_resurrection_inst_unwa",
        "BS Roformer Instrumental Resurrection by Unwa",
        "roformer",
        [
            (
                "bs_resurrection_inst_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_resurrection_inst_unwa.ckpt",
            ),
            (
                "bs_resurrection_inst_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_resurrection_inst_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_resurrection_inst_gabox",
        "BS Roformer Instrumental Resurrection by Gabox",
        "roformer",
        [
            (
                "bs_resurrection_inst_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_resurrection_inst_gabox.ckpt",
            ),
            (
                "bs_resurrection_inst_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_resurrection_inst_gabox_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_karaoke_becruily",
        "BS Roformer Karaoke by Becrily & Frazer",
        "roformer",
        [
            (
                "bs_karaoke_becruily.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_becruily.ckpt",
            ),
            (
                "bs_karaoke_becruily_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_becruily_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST bs roformer model (Instrumental, Vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_voctest_gabox",
        "BS Roformer Vocals by GaboxR67",
        "roformer",
        [
            (
                "bs_voctest_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_voctest_gabox.ckpt",
            ),
            (
                "bs_voctest_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_voctest_gabox_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST bs roformer model (Instrumental, Vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_karaoke_gabox",
        "BS Roformer Karaoke by GaboxR67",
        "roformer",
        [
            (
                "bs_karaoke_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_gabox.ckpt",
            ),
            (
                "bs_karaoke_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_gabox_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_karaoke_inv_gabox",
        "BS Roformer Karaoke Inverted by GaboxR67",
        "roformer",
        [
            (
                "bs_karaoke_inv_gabox.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_inv_gabox.ckpt",
            ),
            (
                "bs_karaoke_inv_gabox_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_inv_gabox_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_6stem_fixed",
        "BS Roformer SW Fixed by jarredou",
        "roformer",
        [
            (
                "bs_6stem_fixed.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed.ckpt",
            ),
            (
                "bs_6stem_fixed_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed_config.yaml",
            ),
        ],
        ["bass", "drums", "guitar", "other", "piano", "vocals"],
        None,
        "MSST bs roformer model (bass, drums, guitar, other, piano, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_logic_6stem",
        "BS Roformer Logic 6 stems by Chantrail",
        "roformer",
        [
            (
                "bs_logic_6stem.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_logic_6stem.ckpt",
            ),
            (
                "bs_logic_6stem_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_logic_6stem_config.yaml",
            ),
        ],
        ["bass", "drums", "guitar", "other", "piano", "vocals"],
        None,
        "MSST bs roformer model (bass, drums, guitar, other, piano, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_4stem_zfturbo",
        "BS Roformer 4 Stems by ZFTurbo",
        "roformer",
        [
            (
                "bs_4stem_zfturbo.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_4stem_zfturbo.ckpt",
            ),
            (
                "bs_4stem_zfturbo_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_4stem_zfturbo_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST bs roformer model (bass, drums, other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_4stemft_syh99999",
        "BS Roformer 4 Stems FT by SYH99999",
        "roformer",
        [
            (
                "bs_4stemft_syh99999.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_4stemft_syh99999.ckpt",
            ),
            (
                "bs_4stemft_syh99999_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_4stemft_syh99999_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST bs roformer model (bass, drums, other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_male_female_146_sucial",
        "BS Roformer Male-Female (ep 146) by Sucial",
        "roformer",
        [
            (
                "bs_male_female_146_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_male_female_146_sucial.ckpt",
            ),
            (
                "bs_male_female_146_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_male_female_146_sucial_config.yaml",
            ),
        ],
        ["female", "male"],
        None,
        "MSST bs roformer model (female, male).",
        cls="bs",
    ),
    UvrModel(
        "bs_male_female_267_sucial",
        "BS Roformer Male-Female (ep 267) by Sucial",
        "roformer",
        [
            (
                "bs_male_female_267_sucial.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_male_female_267_sucial.ckpt",
            ),
            (
                "bs_male_female_267_sucial_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_male_female_267_sucial_config.yaml",
            ),
        ],
        ["female", "male"],
        None,
        "MSST bs roformer model (female, male).",
        cls="bs",
    ),
    UvrModel(
        "bs_male_female_aufr33",
        "BS Roformer Male-Female by Aufr33",
        "roformer",
        [
            (
                "bs_male_female_aufr33.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_male_female_aufr33.ckpt",
            ),
            (
                "bs_male_female_aufr33_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_male_female_aufr33_config.yaml",
            ),
        ],
        ["female", "male"],
        None,
        "MSST bs roformer model (female, male).",
        cls="bs",
    ),
    UvrModel(
        "bs_deverb_256_8_anvuew",
        "BS Roformer Deverb 256-8 by Anvuew",
        "roformer",
        [
            (
                "bs_deverb_256_8_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_deverb_256_8_anvuew.ckpt",
            ),
            (
                "bs_deverb_256_8_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_deverb_256_8_anvuew_config.yaml",
            ),
        ],
        ["noreverb", "reverb"],
        "noreverb",
        "MSST bs roformer model (noreverb, reverb).",
        cls="bs",
    ),
    UvrModel(
        "bs_deverb_384_10_anvuew",
        "BS Roformer Deverb 384-10 by Anvuew",
        "roformer",
        [
            (
                "bs_deverb_384_10_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_deverb_384_10_anvuew.ckpt",
            ),
            (
                "bs_deverb_384_10_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_deverb_384_10_anvuew_config.yaml",
            ),
        ],
        ["noreverb", "reverb"],
        "noreverb",
        "MSST bs roformer model (noreverb, reverb).",
        cls="bs",
    ),
    UvrModel(
        "bs_deverb_room_anvuew",
        "BS Roformer Deverb Roon by Anvuew",
        "roformer",
        [
            (
                "bs_deverb_room_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_deverb_room_anvuew.ckpt",
            ),
            (
                "bs_deverb_room_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_deverb_room_anvuew_config.yaml",
            ),
        ],
        ["noreverb", "reverb"],
        "noreverb",
        "MSST bs roformer model (noreverb, reverb).",
        cls="bs",
    ),
    UvrModel(
        "bs_karaoke_anvuew",
        "BS Roformer Karaoke by Anvuew",
        "roformer",
        [
            (
                "bs_karaoke_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_anvuew.ckpt",
            ),
            (
                "bs_karaoke_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_anvuew_config.yaml",
            ),
        ],
        ["Instrumental", "Vocals"],
        "Vocals",
        "MSST bs roformer model (Instrumental, Vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocals_anvuew",
        "BS Roformer Vocals by Anvuew",
        "roformer",
        [
            (
                "bs_vocals_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_anvuew.ckpt",
            ),
            (
                "bs_vocals_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_anvuew_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "vocals",
        "MSST bs roformer model (instrument, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocalsft1_anvuew",
        "BS Roformer Vocals FT v1 by Anvuew",
        "roformer",
        [
            (
                "bs_vocalsft1_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocalsft1_anvuew.ckpt",
            ),
            (
                "bs_vocalsft1_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocalsft1_anvuew_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "vocals",
        "MSST bs roformer model (instrument, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_mag_anvuew",
        "BS Roformer Mag by Anvuew",
        "roformer",
        [
            (
                "bs_mag_anvuew.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_mag_anvuew.ckpt",
            ),
            (
                "bs_mag_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_mag_anvuew_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "vocals",
        "MSST bs roformer model (instrument, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_4stem_aname",
        "BS Roformer 4 stems by Aname",
        "roformer",
        [
            (
                "bs_4stem_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_4stem_aname.ckpt",
            ),
            (
                "bs_4stem_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_4stem_aname_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST bs roformer model (bass, drums, other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocals1_aname",
        "BS Roformer Vocals v1 by Aname",
        "roformer",
        [
            (
                "bs_vocals1_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals1_aname.ckpt",
            ),
            (
                "bs_vocals_anvuew_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_anvuew_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "vocals",
        "MSST bs roformer model (instrument, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_vocals2_aname",
        "BS Roformer Vocals v2 by Aname",
        "roformer",
        [
            (
                "bs_vocals2_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals2_aname.ckpt",
            ),
            (
                "bs_vocals2_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals2_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_orch_xlancer",
        "BS Roformer Orchestra v1 by Xlance",
        "roformer",
        [
            (
                "bs_orch_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_orch_xlancer.ckpt",
            ),
            (
                "bs_orch_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_orch_xlancer_config.yaml",
            ),
        ],
        ["orch", "other"],
        "orch",
        "MSST bs roformer model (orch, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_orch2_xlancer",
        "BS Roformer Orchestra v2 by Xlance",
        "roformer",
        [
            (
                "bs_orch2_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_orch2_xlancer.ckpt",
            ),
            (
                "bs_orch_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_orch_xlancer_config.yaml",
            ),
        ],
        ["orch", "other"],
        "orch",
        "MSST bs roformer model (orch, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_keys_xlancer",
        "BS Roformer Keys by Xlance",
        "roformer",
        [
            (
                "bs_keys_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_keys_xlancer.ckpt",
            ),
            (
                "bs_keys_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_keys_xlancer_config.yaml",
            ),
        ],
        ["keys", "other"],
        "keys",
        "MSST bs roformer model (keys, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_bass_xlancer",
        "BS Roformer Bass by Xlance",
        "roformer",
        [
            (
                "bs_bass_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_bass_xlancer.ckpt",
            ),
            (
                "bs_bass_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_bass_xlancer_config.yaml",
            ),
        ],
        ["bass", "other"],
        "bass",
        "MSST bs roformer model (bass, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_drums_xlancer",
        "BS Roformer Drums v1 by Xlance",
        "roformer",
        [
            (
                "bs_drums_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_xlancer.ckpt",
            ),
            (
                "bs_drums_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_xlancer_config.yaml",
            ),
        ],
        ["drums", "other"],
        "drums",
        "MSST bs roformer model (drums, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_drums2_xlancer",
        "BS Roformer Drums v2 by Xlance",
        "roformer",
        [
            (
                "bs_drums2_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums2_xlancer.ckpt",
            ),
            (
                "bs_drums2_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums2_xlancer_config.yaml",
            ),
        ],
        ["drums", "other"],
        "drums",
        "MSST bs roformer model (drums, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_gtr_xlancer",
        "BS Guitar by Kimberley Xlance",
        "roformer",
        [
            (
                "bs_gtr_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_gtr_xlancer.ckpt",
            ),
            (
                "bs_gtr_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_gtr_xlancer_config.yaml",
            ),
        ],
        ["guitar", "other"],
        "gtr",
        "MSST bs roformer model (guitar, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_perc_xlancer",
        "BS Roformer Percussion v1 by Xlance",
        "roformer",
        [
            (
                "bs_perc_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_perc_xlancer.ckpt",
            ),
            (
                "bs_perc_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_perc_xlancer_config.yaml",
            ),
        ],
        ["other", "percussion"],
        "percussion",
        "MSST bs roformer model (other, percussion).",
        cls="bs",
    ),
    UvrModel(
        "bs_perc2_xlancer",
        "BS Roformer Percussion v2 by Xlance",
        "roformer",
        [
            (
                "bs_perc2_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_perc2_xlancer.ckpt",
            ),
            (
                "bs_perc2_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_perc2_xlancer_config.yaml",
            ),
        ],
        ["other", "percussion"],
        "percussion",
        "MSST bs roformer model (other, percussion).",
        cls="bs",
    ),
    UvrModel(
        "bs_syn_xlancer",
        "BS Roformer Synth v1 by Xlance",
        "roformer",
        [
            (
                "bs_syn_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_syn_xlancer.ckpt",
            ),
            (
                "bs_syn_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_syn_xlancer_config.yaml",
            ),
        ],
        ["other", "synth"],
        "synth",
        "MSST bs roformer model (other, synth).",
        cls="bs",
    ),
    UvrModel(
        "bs_syn2_xlancer",
        "BS Roformer Synth v2 by Xlance",
        "roformer",
        [
            (
                "bs_syn2_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_syn2_xlancer.ckpt",
            ),
            (
                "bs_syn2_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_syn2_xlancer_config.yaml",
            ),
        ],
        ["other", "synth"],
        "synth",
        "MSST bs roformer model (other, synth).",
        cls="bs",
    ),
    UvrModel(
        "bs_vox_xlancer",
        "BS Roformer Vox by Xlance",
        "roformer",
        [
            (
                "bs_vox_xlancer.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vox_xlancer.ckpt",
            ),
            (
                "bs_vox_xlancer_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vox_xlancer_config.yaml",
            ),
        ],
        ["other", "vox"],
        "vox",
        "MSST bs roformer model (other, vox).",
        cls="bs",
    ),
    UvrModel(
        "bs_drums_gilliaaan",
        "BS Roformer Drums Duality by Gilliaaan",
        "roformer",
        [
            (
                "bs_drums_gilliaaan.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_gilliaaan.ckpt",
            ),
            (
                "bs_drums_gilliaaan_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_drums_gilliaaan_config.yaml",
            ),
        ],
        ["drums", "other"],
        None,
        "MSST bs roformer model (drums, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_mid_side1_gilliaaan",
        "BS Roformer Mid-Side v1 by Gilliaaan",
        "roformer",
        [
            (
                "bs_mid_side1_gilliaaan.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_mid_side1_gilliaaan.ckpt",
            ),
            (
                "bs_mid_side1_gilliaaan_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_mid_side1_gilliaaan_config.yaml",
            ),
        ],
        ["center", "wide"],
        "center",
        "MSST bs roformer model (center, wide).",
        cls="bs",
    ),
    UvrModel(
        "bs_mid_side2_gilliaaan",
        "BS Roformer Mid-Side v2 by Gilliaaan",
        "roformer",
        [
            (
                "bs_mid_side2_gilliaaan.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_mid_side2_gilliaaan.ckpt",
            ),
            (
                "bs_mid_side2_gilliaaan_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_mid_side2_gilliaaan_config.yaml",
            ),
        ],
        ["center", "wide"],
        "center",
        "MSST bs roformer model (center, wide).",
        cls="bs",
    ),
    UvrModel(
        "bs_speech_alicen",
        "BS Roformer SppechSep by AliceN",
        "roformer",
        [
            (
                "bs_speech_alicen.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_speech_alicen.ckpt",
            ),
            (
                "bs_speech_alicen_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_speech_alicen_config.yaml",
            ),
        ],
        ["vocals", "other"],
        "vocals",
        "MSST bs roformer model (vocals, other).",
        cls="bs",
    ),
    UvrModel(
        "bs_pope_4stem_aname",
        "BS PolarFormer 4 stems Lazy Bat by Aname",
        "roformer",
        [
            (
                "bs_pope_4stem_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_pope_4stem_aname.ckpt",
            ),
            (
                "bs_pope_4stem_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_pope_4stem_aname_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST bs roformer model (bass, drums, other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_pope_instvoc_aname",
        "BS PolarFormer InstVoc Duality Lazy Bat by Aname",
        "roformer",
        [
            (
                "bs_pope_instvoc_aname.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_pope_instvoc_aname.ckpt",
            ),
            (
                "bs_pope_instvoc_aname_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_pope_instvoc_aname_config.yaml",
            ),
        ],
        ["other", "vocals"],
        None,
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    UvrModel(
        "bs_pope_vocals_zfturbo",
        "BS PolarFormer Vocals by ZFTurbo",
        "roformer",
        [
            (
                "bs_pope_vocals_zfturbo.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_pope_vocals_zfturbo.ckpt",
            ),
            (
                "bs_pope_vocals_zfturbo_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_pope_vocals_zfturbo_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST bs roformer model (other, vocals).",
        cls="bs",
    ),
    # --- MSST derived variants (own inference scripts, see uvr/roformer/) ---
    UvrModel(
        "mbr_wsa",
        "Windowed Sink Attention Mel-Band Roformer Vocals by Smule Labs",
        "roformer",
        [
            (
                "mbr_wsa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_wsa.ckpt",
            ),
            (
                "mbr_wsa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/mel_band_roformer/mbr_wsa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "vocals",
        "MSST Windowed Sink Attention Mel-Band Roformer Vocals by Smule Labs.",
        cls="windowed",
    ),
    UvrModel(
        "bs_cr_4stem_zf_turbo",
        "BS Conformer 4 Stems by ZFTurbo",
        "roformer",
        [
            (
                "bs_cr_4stem_zf_turbo.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_cr_4stem_zf_turbo.ckpt",
            ),
            (
                "bs_cr_4stem_zf_turbo_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_cr_4stem_zf_turbo_config.yaml",
            ),
        ],
        ["bass", "drums", "other", "vocals"],
        None,
        "MSST BS Conformer 4 Stems by ZFTurbo.",
        cls="bs_conformer",
    ),
    UvrModel(
        "bs_siamese_vocals_unwa",
        "BS Siamese Roformer Vocals by Unwa",
        "roformer",
        [
            (
                "bs_siamese_vocals_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_siamese_vocals_unwa.ckpt",
            ),
            (
                "bs_siamese_vocals_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_siamese_vocals_unwa_config.yaml",
            ),
        ],
        ["instrumental", "vocals"],
        "vocals",
        "MSST BS Siamese Roformer Vocals by Unwa.",
        cls="siamese",
    ),
    UvrModel(
        "bs_inst_fno_unwa",
        "BS Roformer Instrumental FNO by Unwa",
        "roformer",
        [
            (
                "bs_inst_fno_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_fno_unwa.ckpt",
            ),
            (
                "bs_inst_fno_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_fno_unwa_config.yaml",
            ),
        ],
        ["other", "vocals"],
        "other",
        "MSST BS Roformer Instrumental FNO by Unwa.",
        cls="fno",
    ),
    UvrModel(
        "bs_inst_large2_unwa",
        "BS Roformer Instrumental Large v2 by Unwa",
        "roformer",
        [
            (
                "bs_inst_large2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_large2_unwa.ckpt",
            ),
            (
                "bs_inst_large2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_large2_unwa_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "instrument",
        "MSST BS Roformer Instrumental Large v2 by Unwa.",
        cls="unwa_inst_large_2",
    ),
    UvrModel(
        "bs_inst_hyperace_unwa",
        "BS Roformer Instrumental HyperACE (finetuned anvuew vocal model) by Unwa",
        "roformer",
        [
            (
                "bs_inst_hyperace_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_hyperace_unwa.ckpt",
            ),
            (
                "bs_inst_hyperace_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_hyperace_unwa_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "instrument",
        "MSST BS Roformer Instrumental HyperACE (finetuned anvuew vocal model) by Unwa.",
        cls="hyperace",
    ),
    UvrModel(
        "bs_inst_hyperace2_unwa",
        "BS Roformer Instrumental HyperACE v2 (finetuned anvuew vocal model) by Unwa",
        "roformer",
        [
            (
                "bs_inst_hyperace2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_hyperace2_unwa.ckpt",
            ),
            (
                "bs_inst_hyperace2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_inst_hyperace2_unwa_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "instrument",
        "MSST BS Roformer Instrumental HyperACE v2 (finetuned anvuew vocal model) by Unwa.",
        cls="hyperace2",
    ),
    UvrModel(
        "bs_voc_hyperace2_unwa",
        "BS Roformer Vocals HyperACE v2 (finetuned anvuew vocal model) by Unwa",
        "roformer",
        [
            (
                "bs_voc_hyperace2_unwa.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_voc_hyperace2_unwa.ckpt",
            ),
            (
                "bs_voc_hyperace2_unwa_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_voc_hyperace2_unwa_config.yaml",
            ),
        ],
        ["instrument", "vocals"],
        "vocals",
        "MSST BS Roformer Vocals HyperACE v2 (finetuned anvuew vocal model) by Unwa.",
        cls="hyperace2",
    ),
    UvrModel(
        "bs_6stem",
        "BS Roformer SW",
        "roformer",
        [
            (
                "bs_6stem.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem.ckpt",
            ),
            (
                "bs_6stem_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_config.yaml",
            ),
        ],
        ["bass", "drums", "guitar", "other", "piano", "vocals"],
        None,
        "MSST BS Roformer SW.",
        cls="sw",
    ),
    UvrModel(
        "bs_karaoke_3stem_giantailab",
        "BS Roformer Karaoke by GiantAILAB",
        "roformer",
        [
            (
                "bs_karaoke_3stem_giantailab.ckpt",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_3stem_giantailab.ckpt",
            ),
            (
                "bs_karaoke_3stem_giantailab_config.yaml",
                "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_karaoke_3stem_giantailab_config.yaml",
            ),
        ],
        ["backing_vocal", "instrumental", "vocals"],
        None,
        "MSST BS Roformer Karaoke by GiantAILAB.",
        cls="conditional",
    ),
]

BY_KEY = {m.key: m for m in MODELS}
BY_FILENAME = {m.files[0][0]: m for m in MODELS}
BY_ALIAS = {a: m for m in MODELS for a in m.aliases}


def _load_custom_models():
    """Merge user models from uvr/custom_models.json (same entry schema)."""
    import os as _os

    path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "custom_models.json"
    )
    if not _os.path.isfile(path):
        return
    import json as _json

    with open(path, encoding="utf-8") as f:
        data = _json.load(f)
    for e in data if isinstance(data, list) else []:
        try:
            m = UvrModel(
                e["key"],
                e.get("label", e["key"]),
                e.get("arch", "roformer"),
                [(fn, url) for fn, url in e.get("files", [])],
                e.get("stems", []),
                e.get("target"),
                e.get("blurb", ""),
                aliases=tuple(e.get("aliases", ())),
                cls=e.get("cls"),
            )
        except Exception:
            continue
        MODELS.append(m)
        BY_KEY[m.key] = m
        BY_FILENAME[m.files[0][0]] = m
        for a in m.aliases:
            BY_ALIAS[a] = m


_load_custom_models()


def resolve(name):
    """Resolve a model key, weight filename or alias to its catalog entry."""
    return BY_KEY.get(name) or BY_FILENAME.get(name) or BY_ALIAS.get(name)
