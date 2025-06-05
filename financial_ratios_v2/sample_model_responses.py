import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_structured_samples(file_path):
    """Load the structured samples from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_model_response(model, tokenizer, question, max_new_tokens=4096):
    """Generate a response from the model for a given question."""
    # Format the prompt for the model with thinking mode enabled
    prompt = f"Question: {question}\n\nThinking: "
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response with thinking mode parameters
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,  # Recommended for thinking mode
            top_p=0.95,       # Recommended for thinking mode
            top_k=20,         # Recommended for thinking mode
            min_p=0,          # Default setting
            do_sample=True,
        )
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the answer part (remove the prompt)
    answer = response[len(prompt):].strip()
    
    return answer

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    input_file = os.path.join(project_root, "testset/financial_ratios/structured_samples.json")
    output_file = os.path.join(project_root, "testset/financial_ratios/structured_samples_with_responses.json")
    
    # Load the structured samples
    print("Loading structured samples...")
    samples = load_structured_samples(input_file)
    
    # Load model and tokenizer
    print("Loading Qwen3-8B model and tokenizer...")
    model_name = "Qwen/Qwen3-8B"  # Using Qwen3-8B as requested
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Process each sample
    print("Generating model responses...")
    for key, sample in samples.items():
        print(f"Processing {key}...")
        question = sample["question"]
        
        # Generate response from the model
        model_response = generate_model_response(model, tokenizer, question)
        
        # Add the model's response to the sample
        sample["qwen_response"] = model_response
    
    # Save the updated samples
    print("Saving updated samples...")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Successfully saved samples with model responses to {output_file}")

if __name__ == "__main__":
    main() 