import gradio as gr
import os
import sys

pid_file_path = os.path.join(os.getcwd(), "rvc", "train", "train_pid.txt")
now_dir = os.getcwd()


def stop_train(model_name: str):
    pid_file_path = os.path.join(now_dir, "logs", model_name, "train_pid.txt")
    try:
        with open(pid_file_path, "r") as pid_file:
            pids = [int(pid) for pid in pid_file.readlines()]
        for pid in pids:
            os.kill(pid, 9)
        os.remove(pid_file_path)
    except:
        pass


def stop_infer():
    pid_file_path = os.path.join(now_dir, "assets", "infer_pid.txt")
    try:
        with open(pid_file_path, "r") as pid_file:
            pids = [int(pid) for pid in pid_file.readlines()]
        for pid in pids:
            os.kill(pid, 9)
        os.remove(pid_file_path)
    except:
        pass


def restart_applio():
    if os.name != "nt":
        os.system("clear")
    else:
        os.system("cls")
    python = sys.executable
    os.execl(python, python, *sys.argv)


from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def restart_tab():
    with gr.Row():
        with gr.Column():
            restart_button = gr.Button(i18n("Restart Applio"))
            restart_button.click(
                fn=restart_applio,
                inputs=[],
                outputs=[],
            )
