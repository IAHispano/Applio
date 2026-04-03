import json
import os
import sys

import gradio as gr
import psutil

now_dir = os.getcwd()


def stop_train(model_name: str):
    if not model_name or model_name == "":
        return

    pid_file_path = os.path.join(now_dir, "logs", model_name, "config.json")
    killed = 0

    try:
        with open(pid_file_path, "r", encoding="utf-8") as pid_file:
            pid_data = json.load(pid_file)
            pids = pid_data.get("process_pids", [])

        with open(pid_file_path, "w", encoding="utf-8") as pid_file:
            pid_data.pop("process_pids", None)
            json.dump(pid_data, pid_file, indent=4)

        for pid in pids:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                killed += 1
            except psutil.NoSuchProcess:
                pass
            except Exception:
                try:
                    os.kill(pid, 9)
                    killed += 1
                except:
                    pass

        # if killed > 0:
        #    gr.Info(f"Training stopped successfully ({killed} process(es) terminated)")
        # else:
        #    gr.Info("No active training processes found")

    except:
        pass


def stop_infer():
    pid_file_path = os.path.join(now_dir, "assets", "infer_pid.txt")
    try:
        with open(pid_file_path, "r") as pid_file:
            pids = [int(pid) for pid in pid_file.readlines() if pid.strip()]

        for pid in pids:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception:
                try:
                    os.kill(pid, 9)
                except:
                    pass

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
