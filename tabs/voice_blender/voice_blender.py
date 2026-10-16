import os, sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from assets.app_shell import step_card
from core import run_model_blender_script

i18n = I18nAuto()


def update_model_fusion(dropbox):
    gr.Info(i18n("Model added. It is now selected in the model path field."))
    return dropbox, None


def voice_blender_tab():
    def _blend_with_toast(*args):
        if not all(str(model).lower().endswith(".pth") for model in args[1:3]):
            message = i18n("Blending requires .pth voice models; ONNX files are not supported.")
            gr.Warning(message)
            return message, None
        gr.Info(i18n("Blending models..."))
        try:
            result = run_model_blender_script(*args)
        except Exception:
            gr.Warning(
                i18n(
                    "An error occurred blending the models. Please check the console logs for more details."
                )
            )
            raise
        message = result[0] if isinstance(result, tuple) else result
        if isinstance(message, str):
            if "error" in message.lower() or "failed" in message.lower():
                gr.Warning(message)
            else:
                gr.Info(message)
        return result

    with step_card(1, "Source Models", i18n, description="Drop files or paste paths for the two models to blend."):
        with gr.Row():
            with gr.Column():
                model_fusion_a_dropbox = gr.File(
                    label=i18n("Drop your first model here or use the browse button"),
                    file_types=[".pth"],
                    type="filepath",
                )
                model_fusion_a = gr.Textbox(
                    label=i18n("Path to first model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter path to model"),
                    info=i18n("You can also use a custom path."),
                )
            with gr.Column():
                model_fusion_b_dropbox = gr.File(
                    label=i18n("Drop your second model here or use the browse button"),
                    file_types=[".pth"],
                    type="filepath",
                )
                model_fusion_b = gr.Textbox(
                    label=i18n("Path to second model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter path to model"),
                    info=i18n("You can also use a custom path."),
                )
    with step_card(2, "Blend Settings", i18n, description="Name the output and set the blend ratio between models."):
        model_fusion_name = gr.Textbox(
            label=i18n("Model Name"),
            info=i18n("Name of the new model."),
            value="",
            max_lines=1,
            interactive=True,
            placeholder=i18n("Enter model name"),
        )
        alpha_a = gr.Slider(
            minimum=0,
            maximum=1,
            label=i18n("Blend Ratio"),
            value=0.5,
            interactive=True,
            info=i18n(
                "Mix model weights: 0 uses only the second model, 1 uses only the first model. Both models must be compatible."
            ),
        )
    with step_card(3, "Blend & Export", i18n, description="Fuse the models and download the result."):
        model_fusion_button = gr.Button(i18n("Fusion"), variant="primary")
        with gr.Row(equal_height=True, elem_classes="applio-output-row"):
            model_fusion_output_info = gr.Textbox(
                label=i18n("Blend output"),
                info=i18n("The output information will be displayed here."),
                value="",
                lines=3,
            )
            model_fusion_pth_output = gr.File(
                label=i18n("Download Model"), type="filepath", interactive=False
            )

    model_fusion_button.click(
        fn=_blend_with_toast,
        inputs=[
            model_fusion_name,
            model_fusion_a,
            model_fusion_b,
            alpha_a,
        ],
        outputs=[model_fusion_output_info, model_fusion_pth_output],
    )

    model_fusion_a_dropbox.upload(
        fn=update_model_fusion,
        inputs=model_fusion_a_dropbox,
        outputs=[model_fusion_a, model_fusion_a_dropbox],
    )

    model_fusion_b_dropbox.upload(
        fn=update_model_fusion,
        inputs=model_fusion_b_dropbox,
        outputs=[model_fusion_b, model_fusion_b_dropbox],
    )
