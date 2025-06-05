import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_samples(file_path):
    """Load the samples from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def sample_responses(samples, model_name, output_file, device="cuda"):
    """
    Sample responses from a specified model for each problem.
    
    Args:
        samples: Dictionary of samples
        model_name: Name of the model to use
        output_file: Path to save the responses
        device: Device to run the model on
    """
    # Load the model and tokenizer
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    # Prepare to store all responses
    all_responses = {}
    
    # Process each sample
    for key, sample in samples.items():
        print(f"Processing {key}...")
        
        # Create the prompt
        question = sample["question"]
        prompt = f"Solve the following financial problem step by step:\n\n{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        
        # Generate response with thinking mode
        print(f"Generating response for {key}...")
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Set generation parameters
        gen_params = {
            "max_new_tokens": 4096,  # Increased from 1024 to 4096
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Generate response
        with torch.no_grad():
            output = model.generate(input_ids, **gen_params)
        
        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the model's response (remove the prompt)
        model_response = response[len(prompt):]
        
        # Store the response
        all_responses[key] = {
            "question": question,
            "response": model_response,
            "ground_truth": sample["original_answer"],
            "steps": sample["steps"]
        }
        
        print(f"Completed {key}")
    
    # Save all responses to a file
    with open(output_file, 'w') as f:
        json.dump(all_responses, f, indent=2)
    
    print(f"All responses saved to {output_file}")

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    input_file = os.path.join(project_root, "testset/financial_ratios/structured_samples.json")
    output_file = os.path.join(project_root, "testset/financial_ratios/structured_samples_with_responses.json")
    
    # Use Qwen3-8B (Chinese model, advanced reasoning)
    model_name = "Qwen/Qwen3-8B"
    
    # Load the samples
    print("Loading samples...")
    samples = load_samples(input_file)
    
    # Sample responses
    print("Sampling responses...")
    start_time = time.time()
    sample_responses(samples, model_name, output_file)
    end_time = time.time()
    
    print(f"Response sampling completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 