import gradio as gr
import shutil
import os
from src.pipeline import run_pipeline

def process_query(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF file.", ""
    
    if not question.strip():
        return "Please enter a question.", ""
    
    # copy uploaded file to docs folder
    os.makedirs("docs", exist_ok=True)
    shutil.copy(pdf_file.name, "docs/uploaded.pdf")
    
    result = run_pipeline(question, docs_folder="docs")
    
    answer = result["answer"]
    
    sources = "\n\n".join([
        f"Source: {s['source']} | Page: {s['page_number']}\n{s['chunk'][:200]}..."
        for s in result["sources"]
    ])
    
    return answer, sources


with gr.Blocks(title="veridoc") as app:
    gr.Markdown("# veridoc\nAsk questions about your documents. Answers are grounded in your content.")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        question_input = gr.Textbox(label="Your Question", placeholder="What is this document about?")
    
    submit_btn = gr.Button("Get Answer", variant="primary")
    
    answer_output = gr.Textbox(label="Answer", lines=5)
    sources_output = gr.Textbox(label="Sources Used", lines=8)
    
    submit_btn.click(
        fn=process_query,
        inputs=[pdf_input, question_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    app.launch()