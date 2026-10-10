# Roformer implementations (`uvr/roformer_lib/`)

This directory vendors the model classes (torch-only) so Applio never depends on
external separation packages. `registry.py` maps a catalog `cls` value to
`(module, class)` — the same dispatch mvsepless' loader does by config keys:

| `cls`             | class                  | covers                                   |
| ----------------- | ---------------------- | ---------------------------------------- |
| `mel_band`        | `MelBandRoformer`      | standard Mel-Band Roformer models        |
| `bs`              | `BSRoformer`           | standard BS-Roformer models              |
| `windowed`        | `MelBandRoformerWSA`   | windowed-sink-attention variants         |
| `mel_conformer`   | `MelBandConformer`     | mel-band conformer variants              |
| `bs_conformer`    | `BSConformer`          | BS conformer variants                    |
| `sw`              | `BSRoformer_SW`        | sliding-window variants                  |
| `fno`             | `BSRoformer_FNO`       | FNO variants                             |
| `hyperace`        | `BSRoformerHyperACE`   | HyperACE variants                        |
| `hyperace2`       | `BSRoformerHyperACE_2` | HyperACE v2 variants (own `bs_roformer.py`, e.g. Unwa's — incompatible with the stock file, hence its own module) |
| `conditional`     | `BSRoformer_Conditional` | conditional variants                   |
| `siamese`         | `BSSiameseRoformer`    | siamese variants                         |
| `unwa_inst_large_2` | `BSRoformer_2`       | Unwa inst-large-2 variants               |

## Adding a model with a custom architecture

1. Drop its architecture file(s) in `uvr/roformer/custom/` as a module,
   e.g. `my_arch.py`, exposing:
   - `MODEL_CLASS`: the `nn.Module`. Its forward must take stereo waveforms
     `(batch, channels, samples)` and return stems.
   - `VARIANT_KEY = "myarch"` (optional; defaults to the file stem).
   - Use absolute imports (`from uvr.roformer.impl.attend import ...`) or
     relative imports inside a subpackage — never `models.*` paths (that is
     what breaks third-party `bs_roformer.py` files elsewhere).
2. Add a catalog entry (same schema as `uvr/models.py`) to
   `uvr/custom_models.json` with `"cls": "myarch"` plus the checkpoint and
   yaml URLs. No code changes needed; it appears in the UI on next load.
