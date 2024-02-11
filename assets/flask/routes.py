import os, sys
import signal
from flask import Flask, request, redirect

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_download_script

app = Flask(__name__)


@app.route("/download/<path:url>", methods=["GET"])
def download(url):
    file_path = run_download_script(url)
    if file_path == "Model downloaded successfully.":
        if "text/html" in request.headers.get("Accept", ""):
            return redirect("https://applio.org/models/downloaded", code=302)
        else:
            return ""
    else:
        return "Error: Unable to download file", 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    print("This Flask server is shutting down... Please close the window!")
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    app.run(host="localhost", port=8000)
