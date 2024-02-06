import os, sys
import gradio as gr
import importlib.util
import tabs.plugins.plugins_core as plugins_core

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

plugins_core.check_new_folders()


def plugins_tab():
    with gr.TabItem(i18n("Plugin Installer")):
        dropbox = gr.File(
            label=i18n("Drag your plugin.zip to install it"),
            type="filepath",
        )

        dropbox.upload(
            fn=plugins_core.save_plugin_dropbox,
            inputs=[dropbox],
            outputs=[dropbox],
        )

    for plugin in os.listdir(os.path.join(now_dir, "tabs", "plugins", "installed")):
        plugin_main = f"tabs.plugins.installed.{plugin}.plugin"
        plugin_import = importlib.import_module(plugin_main)

        with gr.TabItem(plugin):
            plugin_import.applio_plugin()
