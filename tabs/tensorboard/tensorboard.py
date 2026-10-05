import html
from urllib.parse import urlencode

import gradio as gr

from rvc.lib.tools.launch_tensorboard import available_runs, launch_tensorboard


def open_dashboard(run, view):
    if not run:
        return "", "Select a training run first."
    url = launch_tensorboard(run)
    if url.startswith("Error"):
        return url, html.escape(url)
    if view == "Listening samples":
        url += "#audio"
    elif view == "Learning curves":
        # Time Series filters cards; the older Scalars tab duplicates matched cards.
        query = urlencode(
            {
                "tagFilter": r"^(train/loss|validation/mel_l1|loss/g/total|loss/g/mel|loss/d/adv)$"
            }
        )
        url += "?" + query + "#timeseries"
    else:
        url += "#timeseries"
    return url, (
        f'<iframe src="{html.escape(url, quote=True)}" width="100%" '
        'height="800" frameborder="0" title="Training dashboard"></iframe>'
    )


def tensorboard_tab():
    with gr.Column():
        gr.Markdown(
            "### Training curves and listening samples\n"
            "Choose one run to keep the dashboard focused. **Curves:** compare training loss "
            "and validation mel error within each stage; lower validation error is better. "
            "**Listening samples:** compare the reference with the generated audio at saved updates. "
            "Use **All diagnostics** for gradient norms and optimizer details. "
            "Audio appears after vocoder validation or a saved evaluation. Acoustic stages "
            "predict mel features and need a matching vocoder to produce listening samples."
        )
        runs = available_runs()
        with gr.Row():
            run = gr.Dropdown(
                runs, value=runs[0] if runs else None, label="Training run"
            )
            view = gr.Radio(
                ["Learning curves", "Listening samples", "All diagnostics"],
                value="Learning curves",
                label="View",
            )
            refresh = gr.Button("Refresh runs")
        launch_btn = gr.Button("Open dashboard", variant="primary")
        tb_url = gr.Textbox(label="Open in a browser", interactive=False)
        tb_iframe = gr.HTML("Select a run and open its dashboard.")
        launch_btn.click(open_dashboard, [run, view], [tb_url, tb_iframe])
        view.change(open_dashboard, [run, view], [tb_url, tb_iframe])
        run.change(open_dashboard, [run, view], [tb_url, tb_iframe])
        refresh.click(lambda: gr.update(choices=available_runs()), outputs=run)
