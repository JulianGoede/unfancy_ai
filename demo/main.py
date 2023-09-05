import gradio as gr
import llm


def main():
    with gr.Blocks() as app:
        llm.attach(app)
    app.queue()  # required for streaming output
    app.launch()


if __name__ == "__main__":
    main()
