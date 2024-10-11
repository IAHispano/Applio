import gradio as gr
from rvc.configs.config import Config
from assets.i18n.i18n import I18nAuto

config = Config()
i18n = I18nAuto()

def update_model_settings(
    learning_rate,
    p_dropout,
    use_spectral_norm,
    precision
):
    try:
        if precision not in ["fp16", "fp32"]:
            raise ValueError(f"Invalid precision type: {precision}. Must be 'fp32' or 'fp16'.")

        prev_values = {
            "Learning Rate": config.get_learning_rate(),
            "Dropout": config.get_p_dropout(),
            "Spectral Norm": config.get_use_spectral_norm(),
            "Precision": config.get_precision()
        }

        # Update the settings
        config.set_learning_rate(float(learning_rate))
        config.set_p_dropout(float(p_dropout))
        config.set_use_spectral_norm(use_spectral_norm)
        config.set_precision(precision)

        changed_settings = []
        new_values = {
            "Learning Rate": float(learning_rate),
            "Dropout": float(p_dropout),
            "Spectral Norm": use_spectral_norm,
            "Precision": precision
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

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def update_input_fields():
    return [
        gr.update(interactive=True, value=str(config.get_learning_rate())),
        gr.update(interactive=True, value=str(config.get_p_dropout())),
        gr.update(interactive=True, value=config.get_use_spectral_norm())
    ]

def adv_tab():
    with gr.Row():
      with gr.Column():
        precision = gr.Radio(
            label=i18n("Precision"),
            info=i18n("Select the precision you want to use for training and inference."),
            choices=["fp16", "fp32"],
            value=config.get_precision(),
            interactive=True,
            )

        learning_rate = gr.Textbox(
                label=i18n("Learning Rate"),
                info=i18n("Increasing this setting will make training faster but will converge slower"),
                value=str(config.get_learning_rate()),
                interactive=True,
            )
            
        p_dropout = gr.Textbox(
                label=i18n("P Dropout Rate"),
                value=str(config.get_p_dropout()),
                interactive=True,
            )
            
        use_spectral_norm = gr.Checkbox(
                label=i18n("Use Spectral Normalization"),
                info=i18n("Enabling this setting will make training more stable, lower computational needs and handle higher learning rates"),
                value=config.get_use_spectral_norm(),
                interactive=True,
            )
            
        output = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The updated settings will be displayed here."),
            value="",
            max_lines=11,
            interactive=False,
        )

        update_button = gr.Button(i18n("Update Settings"))
        update_button.click(
            fn=update_model_settings,
            inputs=[
                learning_rate,
                p_dropout,
                use_spectral_norm,
                precision
            ],
            outputs=[output],
        )
