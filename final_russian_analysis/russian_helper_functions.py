
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import sys
import io

# Set stdout to handle UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_pronoun_representation(model, tokenizer, pronoun: str, context: str = ""):
    """Analyze how a pronoun is represented in the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare input
    full_text = f"{context} {pronoun}".strip()
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    # Get token positions
    pronoun_tokens = tokenizer.tokenize(pronoun)
    pronoun_token_ids = tokenizer.encode(pronoun, add_special_tokens=False)
    
    # Print debug info
    print(f"\nDebug tokenization:")
    print(f"Full text: {full_text}")
    print(f"All tokens: {tokenizer.tokenize(full_text)}")
    print(f"Pronoun tokens: {pronoun_tokens}")
    print(f"Pronoun token IDs: {pronoun_token_ids}")
    print(f"Input IDs: {inputs.input_ids[0].tolist()}")
    
    # Find pronoun positions in the sequence
    positions = []
    input_ids = inputs.input_ids[0].tolist()
    for i in range(len(input_ids) - len(pronoun_token_ids) + 1):
        if all(input_ids[i+j] == pronoun_token_ids[j] for j in range(len(pronoun_token_ids))):
            positions.extend(range(i, i + len(pronoun_token_ids)))
    
    if not positions:
        print(f"Warning: Could not find pronoun positions in sequence")
        return None
    
    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    # Analyze attention patterns
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    
    # Analyze each layer
    layer_analyses = []
    for layer_idx, (layer_attention, layer_hidden) in enumerate(zip(attentions, hidden_states)):
        # Get attention for this layer (shape: [batch, heads, seq_len, seq_len])
        attention = layer_attention[0]  # First batch [heads, seq_len, seq_len]
        hidden = layer_hidden[0]  # First batch [seq_len, hidden_size]
        
        # Analyze attention patterns for each head
        head_analyses = []
        for head_idx in range(attention.size(0)):  # Iterate over heads
            head_attention = attention[head_idx]  # [seq_len, seq_len]
            
            # Calculate attention metrics for this head
            attention_to_pronoun = head_attention[:, positions].mean().item()
            attention_from_pronoun = head_attention[positions, :].mean().item()
            self_attention = head_attention[positions][:, positions].mean().item()
            
            head_analyses.append({
                'head': head_idx,
                'self_attention': self_attention,
                'attention_to_pronoun': attention_to_pronoun,
                'attention_from_pronoun': attention_from_pronoun
            })
        
        # Get pronoun representation
        pronoun_repr = hidden[positions].mean(dim=0)
        
        # Find most attentive heads
        head_scores = [h['attention_to_pronoun'] for h in head_analyses]
        most_attentive_heads = np.argsort(head_scores)[-3:].tolist()  # Top 3 heads
        
        layer_analyses.append({
            'layer': layer_idx,
            'head_analyses': head_analyses,
            'most_attentive_heads': most_attentive_heads,
            'representation_norm': torch.norm(pronoun_repr).item()
        })
    
    return {
        'pronoun': pronoun,
        'context': context,
        'tokenization': {
            'tokens': pronoun_tokens,
            'token_ids': pronoun_token_ids,
            'positions': positions
        },
        'layer_analyses': layer_analyses
    }

def visualize_attention_patterns(analysis, layer_idx=0):
    """Visualize attention patterns for a specific layer"""
    layer_analysis = analysis['layer_analyses'][layer_idx]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Average attention patterns
    head_analyses = layer_analysis['head_analyses']
    metrics = ['self_attention', 'attention_to_pronoun', 'attention_from_pronoun']
    avg_values = [np.mean([h[m] for h in head_analyses]) for m in metrics]
    
    ax1.bar(metrics, avg_values)
    ax1.set_title(f'Layer {layer_idx} Average Attention Patterns')
    ax1.set_ylim(0, 1)
    
    # Plot 2: Most attentive heads
    most_attentive = layer_analysis['most_attentive_heads']
    head_scores = [h['attention_to_pronoun'] for h in head_analyses]
    
    ax2.bar(range(len(head_scores)), head_scores)
    ax2.set_title(f'Layer {layer_idx} Head Attention Scores')
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Attention to Pronoun')
    # Highlight most attentive heads
    for head in most_attentive:
        ax2.bar(head, head_scores[head], color='red')
    
    plt.tight_layout()
    plt.show()

def analyze_grammatical_context(model, tokenizer, pronoun: str, case: str, rumi_data: pd.DataFrame):
    """Analyze how a pronoun is represented in different grammatical contexts"""
    # Find sentences in the dataset that use this pronoun in the given case
    # The first column contains the sentences
    relevant_sentences = rumi_data[rumi_data[case] == pronoun].iloc[:, 0].tolist()
    
    if not relevant_sentences:
        print(f"No sentences found for pronoun '{pronoun}' in case '{case}'")
        return []
    
    context_analyses = []
    for sentence in relevant_sentences:
        if pd.isna(sentence):  # Skip NaN values
            continue
        analysis = analyze_pronoun_representation(model, tokenizer, pronoun, sentence)
        if analysis is not None:  # Skip if tokenization failed
            context_analyses.append({
                'context': sentence,
                'analysis': analysis
            })
    
    return context_analyses

def compare_pronouns(model, tokenizer, pronoun1: str, pronoun2: str, context: str = ""):
    """Compare representations of two pronouns"""
    analysis1 = analyze_pronoun_representation(model, tokenizer, pronoun1, context)
    analysis2 = analyze_pronoun_representation(model, tokenizer, pronoun2, context)
    
    # Compare layer representations
    similarities = {}
    for layer_idx in range(len(analysis1['layer_analyses'])):
        layer1 = analysis1['layer_analyses'][layer_idx]
        layer2 = analysis2['layer_analyses'][layer_idx]
        
        # Compare head attention patterns
        head_similarities = []
        for h1, h2 in zip(layer1['head_analyses'], layer2['head_analyses']):
            # Calculate cosine similarity between attention patterns
            pattern1 = np.array([h1['self_attention'], h1['attention_to_pronoun'], h1['attention_from_pronoun']])
            pattern2 = np.array([h2['self_attention'], h2['attention_to_pronoun'], h2['attention_from_pronoun']])
            similarity = np.dot(pattern1, pattern2) / (np.linalg.norm(pattern1) * np.linalg.norm(pattern2))
            head_similarities.append(similarity)
        
        similarities[f'layer_{layer_idx}'] = {
            'mean_similarity': np.mean(head_similarities),
            'head_similarities': head_similarities
        }
    
    return similarities

def predict_case(model, tokenizer, sentence: str, pronoun: str, possible_cases: dict) -> str:
    """
    Predict which case should be used for a pronoun in a given sentence.
    possible_cases is a dictionary mapping case names to their pronoun forms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create prompts for each case
    case_scores = {}
    for case_name, case_pronoun in possible_cases.items():
        try:
            # Create a prompt with the sentence and the pronoun in this case
            prompt = f"{sentence} {case_pronoun}"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Get the logits for the next token
                logits = outputs.logits[0, -1]
                # Get the probability distribution
                probs = torch.softmax(logits, dim=-1)
                # Get the probability of the next token being a space or punctuation
                # This indicates how natural the completion is
                next_token_probs = probs[tokenizer.encode(" .", add_special_tokens=False)[0]]
                case_scores[case_name] = next_token_probs.item()
        except Exception as e:
            print(f"Error processing case {case_name} for sentence: {sentence}")
            print(f"Error: {str(e)}")
            case_scores[case_name] = 0.0  # Assign low score for failed cases
    
    if not case_scores:
        raise ValueError("No valid cases could be processed")
    
    # Return the case with the highest score
    return max(case_scores.items(), key=lambda x: x[1])[0]

def evaluate_case_prediction(model, tokenizer, rumi_data: pd.DataFrame) -> dict:
    """
    Evaluate the model's ability to predict the correct case for pronouns.
    Returns a dictionary with accuracy metrics and prints both correct and wrong predictions.
    """
    case_columns = {
        'nom': ['nom', 'nom_fem', 'nom_plural'],
        'dat': ['dat', 'dat_fem', 'dat_plural', 'dat_n'],
        'gen': ['gen', 'gen_fem', 'gen_plural'],
        'acc': ['acc', 'acc_fem', 'acc_plural'],
        'ins': ['ins', 'ins_fem', 'ins_plural', 'ins_plural_no_n'],
        'prep': ['prep', 'prep_fem', 'prep_plural']
    }

    total_predictions = 0
    correct_predictions = 0
    case_accuracy = {case: {'correct': 0, 'total': 0} for case in case_columns.keys()}
    prediction_patterns = {case: {pred: 0 for pred in case_columns.keys()} for case in case_columns.keys()}
    case_correct = {case: [] for case in case_columns.keys()}
    case_errors = {case: [] for case in case_columns.keys()}

    print("\nEvaluating case prediction accuracy...\n")

    for idx, row in rumi_data.iterrows():
        try:
            sentence = row['sentence '].strip()
            correct_case = row['correct_case '].strip()
            if pd.isna(sentence) or pd.isna(correct_case):
                continue

            is_starred = '*' in correct_case
            base_case = correct_case.replace('*', '')

            possible_cases = {}
            for case, columns in case_columns.items():
                for col in columns:
                    pronoun = row[col]
                    if pd.notna(pronoun) and pronoun.strip():
                        possible_cases[case] = pronoun.strip()

            if not possible_cases:
                continue

            predicted_case = predict_case(model, tokenizer, sentence, "", possible_cases)
            predicted_pronoun = possible_cases.get(predicted_case, "[UNK]")

            total_predictions += 1
            case_accuracy[base_case]['total'] += 1
            is_correct = (predicted_case == correct_case) or (is_starred and predicted_case == base_case)

            if is_correct:
                correct_predictions += 1
                case_accuracy[base_case]['correct'] += 1
                case_correct[base_case].append((sentence, predicted_case, predicted_pronoun))
            else:
                case_errors[base_case].append((sentence, predicted_case, predicted_pronoun))

            prediction_patterns[base_case][predicted_case] += 1

            if total_predictions % 10 == 0:
                print(f"Processed {total_predictions} examples...")
                print(f"Current accuracy: {correct_predictions / total_predictions:.2%}")

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    case_results = {}
    for case, stats in case_accuracy.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        case_results[case] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }

    print("\n--- Prediction Pattern Breakdown ---")
    for true_case, predictions in prediction_patterns.items():
        if case_accuracy[true_case]['total'] > 0:
            print(f"\nTrue case: {true_case}")
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            for pred_case, count in sorted_predictions:
                if count > 0:
                    percentage = count / case_accuracy[true_case]['total'] * 100
                    print(f"  Predicted {pred_case}: {count} times ({percentage:.1f}%)")

    print("\n--- ✅ Correct Predictions ---")
    for case, items in case_correct.items():
        if items:
            print(f"\nTrue case: {case.upper()}")
            for sentence, pred_case, pred_pronoun in items:
                print(f"  → [{pred_case}]  {sentence} ({pred_pronoun})")

    print("\n--- ❌ Wrong Predictions ---")
    for case, errors in case_errors.items():
        if errors:
            print(f"\nTrue case: {case.upper()}")
            for sentence, pred_case, pred_pronoun in errors:
                print(f"  → [{pred_case}] instead of {case} ← {sentence} ({pred_pronoun})")

    return {
        'overall_accuracy': overall_accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'case_results': case_results,
        'prediction_patterns': prediction_patterns
    }

# Load model and data
print("Loading model and data...")
model_name = "ai-forever/rugpt3small_based_on_gpt2"  # Using RuGPT-3 small model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    low_cpu_mem_usage=True
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load the Russian pronoun dataset
rumi_data = pd.read_csv('russian_mi_dataset.csv', encoding='utf-8')

# Print DataFrame info for debugging
print("\nDataFrame columns:")
print(rumi_data.columns.tolist())
print("\nFirst few rows:")
print(rumi_data.head())

# Example analysis
print("\nAnalyzing pronouns...")

# Analyze a masculine pronoun
print("\nAnalyzing 'он' (he):")
analysis = analyze_pronoun_representation(model, tokenizer, "он", "Я вижу")
if analysis is not None:
    print("\nTokenization:")
    print(analysis['tokenization'])

    # Print analysis for first layer
    layer_analysis = analysis['layer_analyses'][0]
    print("\nLayer 0 Analysis:")
    print(f"Most attentive heads: {layer_analysis['most_attentive_heads']}")
    print("\nTop 3 heads attention patterns:")
    for head in layer_analysis['most_attentive_heads']:
        head_analysis = layer_analysis['head_analyses'][head]
        print(f"\nHead {head}:")
        print(f"  Self attention: {head_analysis['self_attention']:.3f}")
        print(f"  Attention to pronoun: {head_analysis['attention_to_pronoun']:.3f}")
        print(f"  Attention from pronoun: {head_analysis['attention_from_pronoun']:.3f}")

    # Visualize attention patterns for first layer
    print("\nGenerating attention visualization...")
    visualize_attention_patterns(analysis, layer_idx=0)

    # Compare with feminine pronoun
    print("\nComparing 'он' (he) with 'она' (she):")
    similarities = compare_pronouns(model, tokenizer, "он", "она")
    print("\nLayer-wise similarities:")
    for layer, sim in similarities.items():
        print(f"{layer}: {sim['mean_similarity']:.3f}")

    # Analyze grammatical context
    print("\nAnalyzing 'он' in nominative case contexts:")
    context_analyses = analyze_grammatical_context(model, tokenizer, "он", "nom", rumi_data)

    print(f"Found {len(context_analyses)} contexts")
else:
    print("Analysis failed due to tokenization issues")

# After loading the model and data, add:
print("\nEvaluating model's case prediction accuracy...")
evaluation_results = evaluate_case_prediction(model, tokenizer, rumi_data)

print("\nEvaluation Results:")
print(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.2%}")
print(f"Total Predictions: {evaluation_results['total_predictions']}")
print(f"Correct Predictions: {evaluation_results['correct_predictions']}")
print("\nPer-Case Accuracy:")
for case, results in evaluation_results['case_results'].items():
    if results['total'] > 0:  # Only show cases that appear in the data
        print(f"{case}: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")