import gradio as gr


def tutorial_tab():
    # with gr.Tab("TTS WebUI"):
    #     gr.Markdown(
    #         """
    #         # TTS WebUI
    #         ### Coming soon!
    #         """
    #     )
    # with gr.Tab("React UI"):
    #     gr.Markdown(
    #         """
    #         # React UI
    #         ### Coming soon!
    #         """
    #     )
    with gr.Tab("Extensions"):
        youtube_video("https://www.youtube.com/watch?v=nfZEoXOGX5Y")
    # with gr.Tab("TTS WebUI scroll through preview 2025 (silent)"):
    #     youtube_video("https://www.youtube.com/watch?v=SSmxO-ccCXE")
    with gr.Tab("ACE-Step Demo"):
        youtube_video("HFtrCnczZQI")
    with gr.Tab("Bark Voice Generation"):
        youtube_video("4LqoEy3BM_0")


def youtube_video(video_id, width="100%", height="500px"):
    if video_id.startswith("https://"):
        video_id = video_id.split("v=")[1]
    gr.HTML(
        f"""<iframe width="{width}" height="{height}" src="https://www.youtube.com/embed/{video_id}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>"""
    )