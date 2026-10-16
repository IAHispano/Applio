import gradio as gr

from rvc.lib.terminal import get_capture

TERMINAL_CSS = """
#applio-terminal textarea {
    font-family: ui-monospace, Consolas, monospace;
    background: #101014;
    color: #e4e4e7;
    font-size: 13px;
    line-height: 1.5;
}
"""


def terminal_tab():
    capture = get_capture()
    start = gr.State(0)
    last_position = gr.State(-1)
    gr.Markdown(
        "Live output from this Applio session, including training and extraction workers."
    )
    with gr.Row():
        live = gr.Checkbox(value=True, label="Live updates")
        clear = gr.Button("Clear view", size="sm")
    output = gr.Textbox(
        label="Terminal",
        value=lambda: capture.read(),
        lines=24,
        max_lines=24,
        interactive=False,
        autoscroll=True,
        elem_id="applio-terminal",
    )
    timer = gr.Timer(0.5)

    def refresh(cursor, previous):
        position = capture.position()
        if position == previous:
            return gr.skip(), gr.skip()
        return capture.read(cursor), position

    def clear_view():
        position = capture.position()
        return position, position, ""

    timer.tick(
        refresh,
        inputs=[start, last_position],
        outputs=[output, last_position],
        queue=False,
        show_progress="hidden",
    )
    live.change(
        lambda active: gr.Timer(active=active), inputs=live, outputs=timer, queue=False
    )
    clear.click(clear_view, outputs=[start, last_position, output], queue=False)
