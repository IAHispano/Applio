import gradio as gr

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def login_tab():
    gr.Markdown(
        value=i18n(
            "## Login"
        )
    )
    gr.Markdown(
        value=i18n(
            "Login with Applio to be able to save information about your models. We will never collect personal information."
        )
    )
    
    gr.Button(i18n("Login with Discord"))
    gr.Button(i18n("Login with Google"))
    gr.Button(i18n("Login with GitHub"))
