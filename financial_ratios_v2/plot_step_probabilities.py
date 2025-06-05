import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.ndimage import gaussian_filter1d

def load_samples_with_probabilities(file_path):
    """Load the samples with calculated probabilities from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_model_responses(file_path):
    """Load the model responses from the JSON file to identify intermediate answers."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_intermediate_answer_positions(response, tokenizer):
    """
    Extract positions where the model provides intermediate answers (before revising them).
    
    Args:
        response: The model's response text
        tokenizer: The tokenizer used for the model
        
    Returns:
        List of token positions where intermediate answers are found
    """
    # Tokenize the response to get token positions
    tokens = tokenizer.encode(response)
    response_text = tokenizer.decode(tokens)
    
    # Define patterns to look for intermediate answers
    # This might include numbers, mathematical expressions, or specific phrases
    # We'll focus on finding calculations that look like "= X" where X is a number
    intermediate_patterns = [
        r"= \d+\.\d+",  # Pattern for = X.XX (decimal numbers)
        r"= \d+",       # Pattern for = X (integers)
        r"\$\d+\.\d+",  # Pattern for $X.XX (dollar amounts)
        r"\$\d+",       # Pattern for $X (dollar amounts)
    ]
    
    positions = []
    for pattern in intermediate_patterns:
        for match in re.finditer(pattern, response_text):
            # Find token position for the match
            match_start = match.start()
            match_text = match.group()
            
            # Count tokens up to this position
            token_position = len(tokenizer.encode(response_text[:match_start]))
            positions.append(token_position)
    
    return positions

def calculate_ema(data, alpha=0.2):
    """
    Calculate Exponential Moving Average for smoothing.
    
    Args:
        data: List of values to smooth
        alpha: Smoothing factor (0 < alpha < 1)
               Lower values = more smoothing
               
    Returns:
        List of smoothed values
    """
    result = [data[0]]  # Start with the first value
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[i-1])
    return result

def plot_step_probabilities(sample_key, sample_data, model_responses, output_dir, smoothing_alpha=0.2, smoothing_sigma=2, tokenizer=None):
    """
    Plot step probabilities against token positions for a given sample.
    
    Args:
        sample_key: The key/name of the sample
        sample_data: The sample data with step probabilities
        model_responses: The model's responses to identify intermediate answers
        output_dir: Directory to save the plot
        smoothing_alpha: Alpha parameter for EMA smoothing
        smoothing_sigma: Sigma parameter for Gaussian smoothing
        tokenizer: Tokenizer to use for identifying token positions
    """
    # Create a figure with a reasonable size
    plt.figure(figsize=(12, 8))
    
    # Color map for different steps
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Plot each step's probabilities
    for i, (step_key, step_data) in enumerate(sample_data["stepwise_probs"].items()):
        # Extract step name from the steps data
        step_name, step_value = sample_data["steps"][i]
        
        # Get positions and probabilities
        positions, probabilities = zip(*step_data)
        
        # Create arrays for smoother interpolation
        positions = np.array(positions)
        probabilities = np.array(probabilities)
        
        # Calculate EMA for smoothing
        smoothed_probs_ema = calculate_ema(probabilities, alpha=smoothing_alpha)
        
        # Also try Gaussian filter for comparison
        smoothed_probs_gaussian = gaussian_filter1d(probabilities, sigma=smoothing_sigma)
        
        # Plot original data points with transparency
        plt.scatter(positions, probabilities, color=colors[i % len(colors)], alpha=0.3, s=30)
        
        # Plot the smoothed line
        plt.plot(positions, smoothed_probs_ema, linestyle='-', linewidth=2, color=colors[i % len(colors)], 
                 label=f"{step_key}: {step_name} = {step_value}")
    
    # Extract intermediate answer positions if we have the tokenizer
    if tokenizer is not None and sample_key in model_responses:
        response_text = model_responses[sample_key]["response"]
        intermediate_positions = extract_intermediate_answer_positions(response_text, tokenizer)
        
        # Plot vertical lines at intermediate answer positions
        for pos in intermediate_positions:
            plt.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)
    
    # Set plot title and labels
    plt.title(f"Step Probabilities for {sample_key.replace('_', ' ').title()}")
    plt.xlabel("Token Position")
    plt.ylabel("Probability of True")
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='best')
    
    # Ensure the y-axis starts from 0 and goes to 1 (for probabilities)
    plt.ylim(0, 1)
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{sample_key}_probabilities.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_file}")

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    input_file = os.path.join(project_root, "testset/financial_ratios/structured_samples_with_probabilities.json")
    responses_file = os.path.join(project_root, "testset/financial_ratios/structured_samples_with_responses.json")
    output_dir = os.path.join(project_root, "testset/financial_ratios/plots")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot step probabilities with smoothing')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for EMA smoothing (0-1)')
    parser.add_argument('--sigma', type=float, default=2, help='Sigma value for Gaussian smoothing')
    args = parser.parse_args()
    
    # Load the samples with probabilities
    print("Loading samples with probabilities...")
    samples = load_samples_with_probabilities(input_file)
    
    # Load model responses to identify intermediate answer positions
    print("Loading model responses...")
    model_responses = load_model_responses(responses_file)
    
    # Try to load the tokenizer used for generating responses
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        print("Loaded tokenizer for identifying intermediate answer positions")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Will not be able to mark intermediate answer positions")
    
    # Plot each sample
    print(f"Generating plots with smoothing (alpha={args.alpha}, sigma={args.sigma})...")
    for key, sample in samples.items():
        print(f"Plotting {key}...")
        plot_step_probabilities(key, sample, model_responses, output_dir, 
                               smoothing_alpha=args.alpha, 
                               smoothing_sigma=args.sigma,
                               tokenizer=tokenizer)
    
    # Create a combined plot for all samples
    print("Creating combined plot...")
    plt.figure(figsize=(15, 10))
    
    # Different line styles for different samples
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    # Track all lines for the legend
    lines = []
    labels = []
    
    # Plot each sample with different line styles
    for s_idx, (sample_key, sample_data) in enumerate(samples.items()):
        # Plot each step's probabilities
        for i, (step_key, step_data) in enumerate(sample_data["stepwise_probs"].items()):
            # Extract step name from the steps data
            step_name, step_value = sample_data["steps"][i]
            
            # Get positions and probabilities
            positions, probabilities = zip(*step_data)
            positions = np.array(positions)
            probabilities = np.array(probabilities)
            
            # Normalize token positions to [0, 1] range for comparable plots
            max_position = max(positions)
            normalized_positions = positions / max_position
            
            # Calculate EMA for smoothing
            smoothed_probs = calculate_ema(probabilities, alpha=args.alpha)
            
            # Plot with different color, line style, and marker
            color_idx = i % 8
            color = plt.cm.tab10(color_idx)
            
            # Plot original data points with transparency
            plt.scatter(normalized_positions, probabilities, 
                      color=color, alpha=0.2, s=15,
                      marker=markers[s_idx % len(markers)])
            
            # Plot the smoothed line
            line, = plt.plot(normalized_positions, smoothed_probs, 
                         linestyle=line_styles[s_idx % len(line_styles)],
                         color=color,
                         linewidth=2)
            
            # Add to legend data
            lines.append(line)
            labels.append(f"{sample_key} - {step_key}: {step_name}")
        
        # Add vertical lines for intermediate answers if available
        if tokenizer is not None and sample_key in model_responses:
            response_text = model_responses[sample_key]["response"]
            intermediate_positions = extract_intermediate_answer_positions(response_text, tokenizer)
            
            # Normalize positions
            normalized_intermediate_positions = [pos / max_position for pos in intermediate_positions]
            
            # Plot vertical lines
            for pos in normalized_intermediate_positions:
                plt.axvline(x=pos, color=plt.cm.tab10(s_idx), linestyle=':', alpha=0.5)
    
    # Set plot title and labels
    plt.title("Step Probabilities Across All Problems (EMA Smoothed)")
    plt.xlabel("Normalized Token Position")
    plt.ylabel("Probability of True")
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')
    
    # Ensure the y-axis starts from 0 and goes to 1 (for probabilities)
    plt.ylim(0, 1)
    
    # Save the combined plot
    combined_file = os.path.join(output_dir, "combined_probabilities_smoothed.png")
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined plot to {combined_file}")
    print("All plots generated successfully!")

if __name__ == "__main__":
    main() 