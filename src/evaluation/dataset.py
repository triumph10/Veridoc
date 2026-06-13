from src.generation.llm import generate

def generate_test_dataset(chunks: list[dict], num_questions: int = 20) -> list[dict]:
    """Generate synthetic QA pairs from chunks using the LLM."""
    dataset = []
    
    for chunk in chunks[:num_questions]:
        prompt = f"""Given this text, generate one specific question that can be answered from it.
        
Text: {chunk['chunk']}

Generate only the question, nothing else."""
        
        question = generate(prompt).strip()
        
        dataset.append({
            "question": question,
            "context": chunk["chunk"],
            "source": chunk["source"],
            "page_number": chunk["page_number"]
        })
    
    return dataset