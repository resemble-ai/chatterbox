import gradio as gr


def get_theme(config):
    default_theme_settings = {
        "theme": "Base",
        "hues": {
            # "primary_hue": "blue",
            "primary_hue": "sky",
            "secondary_hue": "sky",
            "neutral_hue": "neutral",
        },
        "options": {
            "embed_radius": "*radius_sm",
            "block_label_radius": "*radius_sm",
            "block_label_right_radius": "*radius_sm",
            "block_radius": "*radius_sm",
            "block_title_radius": "*radius_sm",
            "container_radius": "*radius_sm",
            "checkbox_border_radius": "*radius_sm",
            "input_radius": "*radius_sm",
            "table_radius": "*radius_sm",
            "button_large_radius": "*radius_sm",
            "button_small_radius": "*radius_sm",
            "button_primary_background_fill_hover": "*primary_300",
            "button_primary_background_fill_hover_dark": "*primary_600",
            "button_secondary_background_fill_hover": "*secondary_200",
            "button_secondary_background_fill_hover_dark": "*secondary_600",
        },
    }
    theme_settings = config.get("theme_settings", default_theme_settings)
    themes = {
        "Base": gr.themes.Base,
        "Default": gr.themes.Default,
        "Monochrome": gr.themes.Monochrome,
    }
    theme: gr.themes.Base = themes[theme_settings["theme"]](
        **theme_settings["hues"],
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
    )
    theme.set(**theme_settings["options"])
    return theme
