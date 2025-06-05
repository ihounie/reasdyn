import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

def load_samples_with_responses(file_path):
    """Load the samples with model responses from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_true_false_probabilities_efficient(model, tokenizer, input_ids, prefix_length, step_name, step_value):
    """
    Calculate the probability of "True" vs "False" for a given step value using KV caching.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: The cached input ids of the response
        prefix_length: Length to truncate the input_ids
        step_name: The name of the step (e.g., "Asset Turnover Ratio")
        step_value: The value of the step
        
    Returns:
        Probability of "True" after softmax between True and False tokens
    """
    # Create the question part for the true/false question
    question = f"\n\nIs {step_name} = {step_value} (Answer with True/False)?"
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    
    # Combine truncated prefix with question
    truncated_input_ids = input_ids[:, :prefix_length]
    question_tensor = torch.tensor([question_ids], device=input_ids.device)
    combined_input_ids = torch.cat([truncated_input_ids, question_tensor], dim=1)
    
    # Get the model's attention mask
    attention_mask = torch.ones(combined_input_ids.shape, device=input_ids.device)
    
    # Run inference with the combined input
    with torch.no_grad():
        outputs = model(combined_input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Get logits for the final token
    
    # Get token IDs for "True" and "False"
    true_token_id = tokenizer.encode(" True")[0]  # Using space prefix for better tokenization
    false_token_id = tokenizer.encode(" False")[0]
    
    # Extract logits for True and False tokens
    true_logit = logits[0, true_token_id].item()
    false_logit = logits[0, false_token_id].item()
    
    # Apply softmax to get probabilities
    logits_array = np.array([true_logit, false_logit])
    probs = np.exp(logits_array) / np.sum(np.exp(logits_array))
    
    # Return probability of True
    return probs[0]

def get_stepwise_probabilities_optimized(model, tokenizer, sample, response, sample_key, stride=4, batch_size=5):
    """
    Process a sample with optimized KV caching and configurable stride.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sample: The sample data
        response: The model's response
        sample_key: The key of the current sample (for saving intermediate results)
        stride: Number of tokens to skip between calculations
        batch_size: Number of token positions to process in each batch
        
    Returns:
        Dictionary of stepwise probabilities
    """
    # Tokenize the full response to get tokens
    input_ids = tokenizer.encode(response, return_tensors="pt").to(model.device)
    
    # Initialize stepwise_probs dictionary
    stepwise_probs = {}
    
    # Process each step in the sample
    for i, (step_name, step_value) in enumerate(sample["steps"]):
        step_key = f"step_{i+1}"
        print(f"  Processing {step_key}: {step_name} = {step_value}")
        
        # Initialize probability list for this step
        step_probs = []
        step_positions = []
        
        # Process in batches for efficiency
        total_length = input_ids.size(1)
        # Using stride to sample at regular intervals
        positions = list(range(1, total_length + 1, stride))
        
        for batch_start_idx in tqdm(range(0, len(positions), batch_size), desc=f"  Processing batches for {step_key}"):
            batch_positions = positions[batch_start_idx:batch_start_idx + batch_size]
            
            # Process each position in this batch
            for pos in batch_positions:
                # Calculate probability for this step at position pos
                prob_true = get_true_false_probabilities_efficient(model, tokenizer, input_ids, pos, step_name, step_value)
                step_probs.append(prob_true)
                step_positions.append(pos)
                
            # Save intermediate results after each batch
            probabilities_with_positions = list(zip(step_positions, step_probs))
            stepwise_probs[step_key] = probabilities_with_positions
            
            # Periodically save all results to handle potential interruptions
            sample["stepwise_probs"] = stepwise_probs
            with open(f"temp_{sample_key}_probabilities.json", "w") as f:
                json.dump({sample_key: sample}, f, indent=2)
        
        # Store final probabilities for this step
        stepwise_probs[step_key] = probabilities_with_positions
    
    return stepwise_probs

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    input_file = os.path.join(project_root, "testset/financial_ratios/structured_samples_with_responses.json")
    output_file = os.path.join(project_root, "testset/financial_ratios/structured_samples_with_probabilities.json")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate stepwise probabilities')
    parser.add_argument('--stride', type=int, default=4, help='Stride for token processing')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing')
    args = parser.parse_args()
    
    # Load the samples with responses
    print("Loading samples with model responses...")
    samples = load_samples_with_responses(input_file)
    
    # Load model and tokenizer
    print("Loading Qwen3-8B model and tokenizer...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Process each sample
    print(f"Calculating stepwise probabilities with stride={args.stride}...")
    for key, sample in samples.items():
        print(f"Processing {key}...")
        response = sample["qwen_response"]
        
        # Process this sample with optimization
        sample["stepwise_probs"] = get_stepwise_probabilities_optimized(
            model, tokenizer, sample, response, key, 
            stride=args.stride, batch_size=args.batch_size
        )
        
        # Save after each sample to handle potential interruptions
        print(f"Saving intermediate results for {key}...")
        with open(f"temp_{key}_probabilities.json", "w") as f:
            json.dump({key: sample}, f, indent=2)
    
    # Save the final updated samples
    print("Saving all samples with probabilities...")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Successfully saved samples with stepwise probabilities to {output_file}")

if __name__ == "__main__":
    main() 