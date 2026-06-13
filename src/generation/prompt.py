from jinja2 import Template
from pathlib import Path


def build_prompt(query: str, chunks: list[dict]) -> str:
    template_path = Path("prompts/qa.j2")
    template = Template(template_path.read_text())
    
    
    context = "\n\n".join([
        f"Source: {chunk['source']} (page{chunk['page_number']})\n{chunk['chunk']}"
        for chunk in chunks 
    ])
    
    return template.render(query =  query, context = context)
