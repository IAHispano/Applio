import os, sys
import gradio as gr
import importlib.util
import tabs.plugins.plugins_core as plugins_core

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

plugins_core.check_new_folders()
main_func_name = "applio_plugin"

def plugins_tab():
    with gr.Tab(label=i18n("Plugin Installer")):
        dropbox = gr.File(
            label=i18n("Drag your plugin.zip to install it"),
            type="filepath",
        )

        dropbox.upload(
            fn=plugins_core.save_plugin_dropbox,
            inputs=[dropbox],
            outputs=[dropbox],
        )

    plugins_dir = os.path.join(now_dir, "tabs", "plugins", "installed")

    for plugin in os.listdir(plugins_dir):
        plugin_path = os.path.join(plugins_dir, plugin)
        plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]

        if not os.path.isdir(plugin_path) or plugin.startswith("."):
            continue

        plugin_main = f"tabs.plugins.installed.{plugin}.plugin"

        try:
            plugin_import = importlib.import_module(plugin_main)

            if hasattr(plugin_import, main_func_name):
                with gr.Tab(label=plugin):
                    plugin_import.applio_plugin()
            else:
                raise gr.Error(
                    i18n('Plugin "{plugin_name}" missing {func_name}()').format(
                        plugin_name = plugin_name,
                        func_name = main_func_name
                    )
                )
        except Exception as e:
            raise gr.Error(
                i18n('Error loading plugin "{plugin_name}":').format(
                    plugin_name = plugin_name
                ) + " " + e
            )
