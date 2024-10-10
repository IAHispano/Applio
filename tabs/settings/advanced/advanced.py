import gradio as gr
from rvc.configs.config import Config
from assets.i18n.i18n import I18nAuto

config = Config()
i18n = I18nAuto()

# Presets
MODEL_PRESETS = {
    "Default": {
        "inter": 192,
        "hidden": 192,
        "filter": 768,
        "learning_rate": 1e-4,
        "p_dropout": 0,
        "use_spectral_norm": False,
        "gin_channels": 256,
        "spk_embed_dim": 109
    },
    "2x Default": {
        "inter": 384,
        "hidden": 384,
        "filter": 1536,
        "learning_rate": 5e-5,
        "p_dropout": 0.5,
        "use_spectral_norm": True,
        "gin_channels": 512,
        "spk_embed_dim": 218
    },
    "3x Default": {
        "inter": 576,
        "hidden": 576,
        "filter": 2304,
        "learning_rate": 5e-5,
        "p_dropout": 0.5,
        "use_spectral_norm": True,
        "gin_channels": 768,
        "spk_embed_dim": 327
    }
}

def update_model_settings(
    preset,
    inter_channels,
    hidden_channels,
    filter_channels,
    learning_rate,
    p_dropout,
    use_spectral_norm,
    gin_channels,
    spk_embed_dim,
):
    try:

        # If preset is selected, use preset values
        if preset != "Custom":
            values = MODEL_PRESETS[preset]
            inter_channels = values["inter"]
            hidden_channels = values["hidden"]
            filter_channels = values["filter"]
            learning_rate = values["learning_rate"]
            p_dropout = values["p_dropout"]
            use_spectral_norm = values["use_spectral_norm"]
            gin_channels = values["gin_channels"]
            spk_embed_dim = values["spk_embed_dim"]

        prev_values = {
            "Inter Channels": config.get_inter_channels(),
            "Hidden Channels": config.get_hidden_channels(),
            "Filter Channels": config.get_filter_channels(),
            "Learning Rate": config.get_learning_rate(),
            "Dropout": config.get_p_dropout(),
            "Spectral Norm": config.get_use_spectral_norm(),
            "Gin Channels": config.get_gin_channels(),
            "Speaker Embed Dim": config.get_spk_embed_dim(),
            "Precision": config.get_precision()
        }

        
        # Update all settings
        config.set_inter_channels(inter_channels)
        config.set_hidden_channels(hidden_channels)
        config.set_filter_channels(filter_channels)
        config.set_learning_rate(float(learning_rate))
        config.set_p_dropout(float(p_dropout))
        config.set_use_spectral_norm(use_spectral_norm)
        config.set_gin_channels(int(gin_channels))
        config.set_spk_embed_dim(int(spk_embed_dim))

        
        changed_settings = []
        new_values = {
            "Inter Channels": inter_channels,
            "Hidden Channels": hidden_channels,
            "Filter Channels": filter_channels,
            "Learning Rate": float(learning_rate),
            "Dropout": float(p_dropout),
            "Spectral Norm": use_spectral_norm,
            "Gin Channels": int(gin_channels),
            "Speaker Embed Dim": int(spk_embed_dim)
        }
        for key, new_value in new_values.items():
            if key in prev_values and prev_values[key] != new_value:
                if isinstance(new_value, float) and key == "Learning Rate":
                    # Format learning rate to scientific notation
                    changed_settings.append(f"{key}: {prev_values[key]:.1e} → {new_value:.1e}")
                else:
                    changed_settings.append(f"{key}: {prev_values[key]} → {new_value}")
        
        if not changed_settings:
            return "No settings were changed."
        else:
            return "Changed settings:\n" + "\n".join(changed_settings)

        return f"Settings updated:\n" \
               f"Inter Channels: {inter_channels}\n" \
               f"Hidden Channels: {hidden_channels}\n" \
               f"Filter Channels: {filter_channels}\n" \
               f"Learning Rate: {learning_rate}\n" \
               f"Dropout: {p_dropout}\n" \
               f"Spectral Norm: {use_spectral_norm}\n" \
               f"Gin Channels: {gin_channels}\n" \
               f"Speaker Embed Dim: {spk_embed_dim}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def update_input_fields(preset):
    if preset == "Custom":
        return [
            gr.update(interactive=True, value=config.get_inter_channels()),
            gr.update(interactive=True, value=config.get_hidden_channels()),
            gr.update(interactive=True, value=config.get_filter_channels()),
            gr.update(interactive=True, value=str(config.get_learning_rate())),
            gr.update(interactive=True, value=str(config.get_p_dropout())),
            gr.update(interactive=True, value=config.get_use_spectral_norm()),
            gr.update(interactive=True, value=str(config.get_gin_channels())),
            gr.update(interactive=True, value=str(config.get_spk_embed_dim()))
        ]
    else:
        values = MODEL_PRESETS[preset]
        return [
            gr.update(interactive=False, value=values["inter"]),
            gr.update(interactive=False, value=values["hidden"]),
            gr.update(interactive=False, value=values["filter"]),
            gr.update(interactive=False, value=str(values["learning_rate"])),
            gr.update(interactive=False, value=str(values["p_dropout"])),
            gr.update(interactive=False, value=values["use_spectral_norm"]),
            gr.update(interactive=False, value=str(values["gin_channels"])),
            gr.update(interactive=False, value=str(values["spk_embed_dim"]))
        ]

def adv_tab():
    with gr.Row():
     with gr.Column():
        preset_choices = list(MODEL_PRESETS.keys()) + ["Custom"]
        model_preset = gr.Dropdown(
            label=i18n("Presets"),
            info=i18n("Choose a preset for all model parameters"),
            choices=preset_choices,
            value="Default",
            interactive=True,
        )
        
        with gr.Column() as settings_column:
            inter_channels = gr.Number(
                label=i18n("Inter Channels"),
                info=i18n("Increasing this setting will make training slower and consume more resources"),
                value=MODEL_PRESETS["Default"]["inter"],
                interactive=False,
            )
            
            hidden_channels = gr.Number(
                label=i18n("Hidden Channels"),
                info=i18n("Increasing this setting will make training slower and consume more resources"),
                value=MODEL_PRESETS["Default"]["hidden"],
                interactive=False,
            )
            
            filter_channels = gr.Number(
                label=i18n("Filter Channels"),
                info=i18n("Increasing this setting will make training slower and consume more resources"),
                value=MODEL_PRESETS["Default"]["filter"],
                interactive=False,
            )
            
            learning_rate = gr.Textbox(
                label=i18n("Learning Rate"),
                info=i18n("Increasing this setting will make training faster but will converge slower"),
                value=str(MODEL_PRESETS["Default"]["learning_rate"]),
                interactive=False,
            )
            
            p_dropout = gr.Textbox(
                label=i18n("P Dropout Rate"),
                value=str(MODEL_PRESETS["Default"]["p_dropout"]),
                interactive=False,
            )
            
            use_spectral_norm = gr.Checkbox(
                label=i18n("Use Spectral Normalization"),
                info=i18n("Enabling this setting will make training more stable, lower computational needs and will be able to handle higher learing rates"),
                value=MODEL_PRESETS["Default"]["use_spectral_norm"],
                interactive=False,
            )
            
            gin_channels = gr.Number(
                label=i18n("Gin Channels"),
                info=i18n("Increasing this setting will make training consume more resources"),
                value=MODEL_PRESETS["Default"]["gin_channels"],
                interactive=False,
            )
            
            spk_embed_dim = gr.Number(
                label=i18n("Speaker Embed Dim"),
                info=i18n("Increasing this setting will make training consume more resources"),
                value=MODEL_PRESETS["Default"]["spk_embed_dim"],
                interactive=False,
            )
            
        output = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The updated settings will be displayed here."),
            value="",
            max_lines=11,
            interactive=False,
        )

        model_preset.change(
            fn=update_input_fields,
            inputs=[model_preset],
            outputs=[
                inter_channels,
                hidden_channels,
                filter_channels,
                learning_rate,
                p_dropout,
                use_spectral_norm,
                gin_channels,
                spk_embed_dim
            ]
        )

        update_button = gr.Button(i18n("Update Settings"))
        update_button.click(
            fn=update_model_settings,
            inputs=[
                model_preset,
                inter_channels,
                hidden_channels,
                filter_channels,
                learning_rate,
                p_dropout,
                use_spectral_norm,
                gin_channels,
                spk_embed_dim
                
            ],
            outputs=[output],
        )
