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
    def _plugin_install_with_toast(dropbox):
        gr.Info(i18n("Installing plugin..."))
        try:
            return plugins_core.save_plugin_dropbox(dropbox)
        except gr.Error:
            raise  # gradio announces gr.Error natively
        except Exception:
            gr.Warning(
                i18n(
                    "An error occurred installing the plugin. Please check the console logs for more details."
                )
            )
            raise

    with gr.TabItem(i18n("Plugin Installer")):
        dropbox = gr.File(
            label=i18n("Drop a plugin.zip here or use the browse button to install it"),
            type="filepath",
        )

        dropbox.upload(
            fn=_plugin_install_with_toast,
            inputs=[dropbox],
            outputs=[dropbox],
        )

    for plugin in os.listdir(os.path.join(now_dir, "tabs", "plugins", "installed")):
        plugin_main = f"tabs.plugins.installed.{plugin}.plugin"
        plugin_import = importlib.import_module(plugin_main)

        with gr.TabItem(plugin):
            plugin_import.applio_plugin()
