from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple
import torch

class QAEngine:
    """Question Answering Engine using local LLM"""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize QA model
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading QA model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def generate_answer(self, question: str, context_chunks: List[Tuple[str, float]]) -> str:
        """Generate answer based on question and context"""
        # Combine relevant chunks
        context = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        # Create prompt
        prompt = f"""Answer the question based only on the context below.

Context: {context}

Question: {question}

Answer:"""
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    min_length=10,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Clean answer
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            # Remove any remaining prompt artifacts
            answer = answer.replace(prompt, "").strip()
            
            if len(answer) < 5 or answer.lower() in ["", "none", "null"]:
                return "I cannot find this information in the document."
            
            return answer
        
        except Exception as e:
            return f"Error: {str(e)}"