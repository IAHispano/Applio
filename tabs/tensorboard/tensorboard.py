import gradio as gr
from assets.i18n.i18n import I18nAuto
from rvc.lib.tools.launch_tensorboard import launch_tensorboard

i18n = I18nAuto()


def tensorboard_tab():
    def launch_and_get_url():
        url = launch_tensorboard()
        if url and not url.startswith("Error"):
            return url, f'<iframe src="{url}" width="100%" height="800" frameborder="0"></iframe>'
        return url or "Failed to start", "<p>Failed to launch TensorBoard</p>"

    with gr.Column():
        gr.Markdown(
            i18n("### TensorBoard\nMonitor training metrics in real-time.")
        )
        with gr.Row():
            launch_btn = gr.Button(
                i18n("Launch TensorBoard"), variant="primary"
            )
        tb_url = gr.Textbox(
            label=i18n("TensorBoard URL"),
            value="",
            interactive=False,
            visible=False,
        )
        tb_iframe = gr.HTML(
            value="<p style='color: gray; text-align: center; padding: 40px;'>Click 'Launch TensorBoard' to start monitoring.</p>"
        )

        launch_btn.click(
            fn=launch_and_get_url,
            inputs=[],
            outputs=[tb_url, tb_iframe],
        )
