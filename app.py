from __future__ import annotations
import os
import sys
import gradio as gr
from ui.app import create_ui

if __name__ == "__main__":
    print("=" * 60)
    print("      LAUNCHING SENSEIBOT ROCKET LEAGUE ML STUDIO     ")
    print("=" * 60)

    # Ensure required directories exist
    os.makedirs("config", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    demo = create_ui()
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate"
    )
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True,
            theme=theme
        )
    except OSError:
        # Fallback to automatic free port selection if 7860 is occupied
        print(f"[SenseiBot Studio] Port {port} is occupied. Finding an available port...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=None,
            share=False,
            show_error=True,
            theme=theme
        )
