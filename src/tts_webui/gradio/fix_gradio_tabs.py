import gradio as gr
from gradio_goodtabs import GoodTabs
from gradio_goodtab import GoodTab as Tab


def fix_gradio_tabs():
    gr.Tab = Tab
    gr.Tabs = GoodTabs
