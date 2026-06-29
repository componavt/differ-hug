import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", gr.Slider(value=1, minimum=1, maximum=10, step=1)],
    outputs="text",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)