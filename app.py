import gradio as gr
import shutil
import os
from src.pipeline import run_pipeline


def process_query(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF file first.", ""
    if not question.strip():
        return "Please enter a question.", ""

    os.makedirs("docs", exist_ok=True)
    shutil.copy(pdf_file.name, "docs/uploaded.pdf")

    result = run_pipeline(question, docs_folder="docs")

    answer = result["answer"]
    sources = "\n\n---\n\n".join([
        f"📄 {s['source']} | Page {s['page_number']}\n{s['chunk'][:300]}..."
        for s in result["sources"]
    ])

    return answer, sources


with gr.Blocks(title="veridoc", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📄 veridoc
    **Ask questions about your documents. Every answer is grounded in your content — no hallucination.**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What is this document about?",
                lines=3
            )
            submit_btn = gr.Button("Get Answer", variant="primary", size="lg")

        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer", lines=8)
            sources_output = gr.Textbox(label="Sources Used", lines=10)

    submit_btn.click(
        fn=process_query,
        inputs=[pdf_input, question_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    app.launch()