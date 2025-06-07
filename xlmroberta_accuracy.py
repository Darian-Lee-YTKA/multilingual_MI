import torch
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import sys
import io

# Set stdout to handle UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Russian preposition dictionary with case patterns
prepositions = {
    'в': {'cases': ['prep', 'acc'], 'context': ['location', 'time', 'state']},
    'на': {'cases': ['prep', 'acc'], 'context': ['surface', 'time', 'state']},
    'с': {'cases': ['ins'], 'context': ['accompaniment', 'instrument', 'manner']},
    'к': {'cases': ['dat'], 'context': ['direction', 'purpose']},
    'от': {'cases': ['gen'], 'context': ['source', 'cause', 'separation']},
    'у': {'cases': ['gen'], 'context': ['possession', 'location']},
    'по': {'cases': ['dat', 'prep'], 'context': ['surface', 'time', 'manner']},
    'о': {'cases': ['prep'], 'context': ['topic', 'subject']},
    'за': {'cases': ['acc', 'ins'], 'context': ['location', 'time', 'purpose']},
    'под': {'cases': ['acc', 'ins'], 'context': ['location', 'state']},
    'над': {'cases': ['ins'], 'context': ['location', 'superiority']},
    'перед': {'cases': ['ins'], 'context': ['location', 'time']},
    'между': {'cases': ['ins'], 'context': ['location', 'relationship']},
    'через': {'cases': ['acc'], 'context': ['time', 'space']},
    'без': {'cases': ['gen'], 'context': ['absence', 'lack']},
    'для': {'cases': ['gen'], 'context': ['purpose', 'benefit']},
    'до': {'cases': ['gen'], 'context': ['limit', 'time']},
    'из': {'cases': ['gen'], 'context': ['source', 'material']},
    'про': {'cases': ['acc'], 'context': ['topic', 'subject']},
    'ради': {'cases': ['gen'], 'context': ['purpose', 'benefit']},
    'сквозь': {'cases': ['acc'], 'context': ['space', 'time']},
    'среди': {'cases': ['gen'], 'context': ['location', 'group']},
    'вокруг': {'cases': ['gen'], 'context': ['location', 'surroundings']},
    'вдоль': {'cases': ['gen'], 'context': ['direction', 'location']},
    'внутри': {'cases': ['gen'], 'context': ['location', 'interior']},
    'вне': {'cases': ['gen'], 'context': ['location', 'exclusion']},
    'вопреки': {'cases': ['dat'], 'context': ['opposition', 'contrast']},
    'благодаря': {'cases': ['dat'], 'context': ['cause', 'reason']},
    'согласно': {'cases': ['dat'], 'context': ['agreement', 'correspondence']},
    'навстречу': {'cases': ['dat'], 'context': ['direction', 'movement']},
    'наперекор': {'cases': ['dat'], 'context': ['opposition', 'defiance']},
    'подобно': {'cases': ['dat'], 'context': ['similarity', 'comparison']},
    'соответственно': {'cases': ['dat'], 'context': ['correspondence', 'agreement']}
}

# Russian verb dictionary with case patterns
verb_cases = {
    'видеть': {'cases': ['acc'], 'context': ['perception', 'observation']},
    'любить': {'cases': ['acc'], 'context': ['emotion', 'preference']},
    'ждать': {'cases': ['gen'], 'context': ['expectation', 'anticipation']},
    'бояться': {'cases': ['gen'], 'context': ['emotion', 'fear']},
    'хотеть': {'cases': ['gen'], 'context': ['desire', 'wish']},
    'давать': {'cases': ['dat'], 'context': ['transfer', 'giving']},
    'говорить': {'cases': ['dat', 'prep'], 'context': ['communication', 'speech']},
    'помогать': {'cases': ['dat'], 'context': ['assistance', 'support']},
    'управлять': {'cases': ['ins'], 'context': ['control', 'direction']},
    'гордиться': {'cases': ['ins'], 'context': ['emotion', 'pride']},
    'заниматься': {'cases': ['ins'], 'context': ['activity', 'occupation']},
    'думать': {'cases': ['prep'], 'context': ['thought', 'consideration']},
    'мечтать': {'cases': ['prep'], 'context': ['imagination', 'desire']},
    'встречать': {'cases': ['acc'], 'context': ['meeting', 'greeting']},
    'понимать': {'cases': ['acc'], 'context': ['comprehension', 'understanding']},
    'знать': {'cases': ['acc'], 'context': ['knowledge', 'familiarity']},
    'слушать': {'cases': ['acc'], 'context': ['perception', 'attention']},
    'смотреть': {'cases': ['acc'], 'context': ['perception', 'observation']},
    'чувствовать': {'cases': ['acc'], 'context': ['perception', 'sensation']},
    'желать': {'cases': ['gen'], 'context': ['desire', 'wish']},
    'требовать': {'cases': ['gen'], 'context': ['demand', 'request']},
    'просить': {'cases': ['gen'], 'context': ['request', 'appeal']},
    'лишиться': {'cases': ['gen'], 'context': ['loss', 'deprivation']},
    'добиваться': {'cases': ['gen'], 'context': ['achievement', 'attainment']},
    'добиться': {'cases': ['gen'], 'context': ['achievement', 'success']},
    'избегать': {'cases': ['gen'], 'context': ['avoidance', 'evasion']},
    'избежать': {'cases': ['gen'], 'context': ['avoidance', 'escape']},
    'касаться': {'cases': ['gen'], 'context': ['contact', 'relation']},
    'касаться': {'cases': ['prep'], 'context': ['relation', 'connection']},
    'надеяться': {'cases': ['prep'], 'context': ['hope', 'expectation']},
    'заботиться': {'cases': ['prep'], 'context': ['care', 'concern']},
    'интересоваться': {'cases': ['ins'], 'context': ['interest', 'curiosity']},
    'увлекаться': {'cases': ['ins'], 'context': ['interest', 'enthusiasm']},
    'восхищаться': {'cases': ['ins'], 'context': ['admiration', 'delight']},
    'довольствоваться': {'cases': ['ins'], 'context': ['satisfaction', 'contentment']},
    'пользоваться': {'cases': ['ins'], 'context': ['use', 'utilization']},
    'обладать': {'cases': ['ins'], 'context': ['possession', 'ownership']},
    'владеть': {'cases': ['ins'], 'context': ['possession', 'control']},
    'руководить': {'cases': ['ins'], 'context': ['control', 'direction']},
    'командовать': {'cases': ['ins'], 'context': ['control', 'authority']},
    'распоряжаться': {'cases': ['ins'], 'context': ['control', 'management']}
}

def analyze_pronoun_representation(model, tokenizer, pronoun: str, context: str = ""):
    """Analyze how a pronoun is represented in the model using XLM-RoBERTa"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare input with mask token
    full_text = f"{context} {tokenizer.mask_token}".strip()
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    # Get token positions for the mask token
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    positions = mask_token_index.tolist()
    
    # Print debug info
    print(f"\nDebug tokenization:")
    print(f"Full text: {full_text}")
    print(f"All tokens: {tokenizer.tokenize(full_text)}")
    print(f"Mask token position: {positions}")
    print(f"Input IDs: {inputs.input_ids[0].tolist()}")
    
    if not positions:
        print(f"Warning: Could not find mask token position in sequence")
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
            attention_to_mask = head_attention[:, positions].mean().item()
            attention_from_mask = head_attention[positions, :].mean().item()
            self_attention = head_attention[positions][:, positions].mean().item()
            
            head_analyses.append({
                'head': head_idx,
                'self_attention': self_attention,
                'attention_to_mask': attention_to_mask,
                'attention_from_mask': attention_from_mask
            })
        
        # Get mask token representation
        mask_repr = hidden[positions].mean(dim=0)
        
        # Find most attentive heads
        head_scores = [h['attention_to_mask'] for h in head_analyses]
        most_attentive_heads = np.argsort(head_scores)[-3:].tolist()  # Top 3 heads
        
        layer_analyses.append({
            'layer': layer_idx,
            'head_analyses': head_analyses,
            'most_attentive_heads': most_attentive_heads,
            'representation_norm': torch.norm(mask_repr).item()
        })
    
    return {
        'pronoun': pronoun,
        'context': context,
        'tokenization': {
            'tokens': tokenizer.tokenize(pronoun),
            'token_ids': tokenizer.encode(pronoun, add_special_tokens=False),
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
    metrics = ['self_attention', 'attention_to_mask', 'attention_from_mask']
    avg_values = [np.mean([h[m] for h in head_analyses]) for m in metrics]
    
    ax1.bar(metrics, avg_values)
    ax1.set_title(f'Layer {layer_idx} Average Attention Patterns')
    ax1.set_ylim(0, 1)
    
    # Plot 2: Most attentive heads
    most_attentive = layer_analysis['most_attentive_heads']
    head_scores = [h['attention_to_mask'] for h in head_analyses]
    
    ax2.bar(range(len(head_scores)), head_scores)
    ax2.set_title(f'Layer {layer_idx} Head Attention Scores')
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Attention to Mask')
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
            pattern1 = np.array([h1['self_attention'], h1['attention_to_mask'], h1['attention_from_mask']])
            pattern2 = np.array([h2['self_attention'], h2['attention_to_mask'], h2['attention_from_mask']])
            similarity = np.dot(pattern1, pattern2) / (np.linalg.norm(pattern1) * np.linalg.norm(pattern2))
            head_similarities.append(similarity)
        
        similarities[f'layer_{layer_idx}'] = {
            'mean_similarity': np.mean(head_similarities),
            'head_similarities': head_similarities
        }
    
    return similarities

def predict_case(model, tokenizer, sentence: str, pronoun: str, possible_cases: dict) -> tuple[str, dict[str, float]]:
    """
    Enhanced prediction with improved grammatical pattern recognition and scoring.
    Focuses on better handling of non-nominative cases while maintaining nominative accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Optimized case parameters based on RuGPT performance analysis
    case_params = {
        'nom': {'temperature': 0.7, 'weight': 1.2, 'prompt_weight': 1.5, 'context_boost': 1.3},
        'dat': {'temperature': 0.8, 'weight': 1.3, 'prompt_weight': 1.4, 'context_boost': 1.4},
        'gen': {'temperature': 0.8, 'weight': 1.3, 'prompt_weight': 1.4, 'context_boost': 1.4},
        'acc': {'temperature': 0.7, 'weight': 1.4, 'prompt_weight': 1.5, 'context_boost': 1.5},
        'ins': {'temperature': 0.8, 'weight': 1.3, 'prompt_weight': 1.4, 'context_boost': 1.4},
        'prep': {'temperature': 0.8, 'weight': 1.3, 'prompt_weight': 1.4, 'context_boost': 1.4}
    }
    
    def analyze_context(sentence: str) -> dict:
        """Enhanced context analysis with improved grammatical pattern recognition"""
        words = sentence.lower().split()
        context = {
            'prepositions': [],
            'verbs': [],
            'is_question': '?' in sentence or any(w in ['кто', 'что', 'где', 'когда', 'почему', 'как'] for w in words),
            'has_negation': any(w in ['не', 'нет', 'ни'] for w in words),
            'word_positions': {word: i for i, word in enumerate(words)},
            'sentence_structure': {
                'has_subject': False,
                'has_direct_object': False,
                'has_indirect_object': False,
                'has_prepositional_phrase': False
            }
        }
        
        # Analyze sentence structure and grammatical patterns
        for i, word in enumerate(words):
            if word in prepositions:
                prep_info = prepositions[word]
                context['prepositions'].append({
                    'word': word,
                    'position': i,
                    'cases': prep_info['cases'],
                    'context': prep_info['context'],
                    'is_prepositional': 'prep' in prep_info['cases'],
                    'is_genitive': 'gen' in prep_info['cases']
                })
                if 'prep' in prep_info['cases']:
                    context['sentence_structure']['has_prepositional_phrase'] = True
            
            elif word in verb_cases:
                verb_info = verb_cases[word]
                context['verbs'].append({
                    'word': word,
                    'position': i,
                    'cases': verb_info['cases'],
                    'context': verb_info['context'],
                    'is_transitive': 'acc' in verb_info['cases'],
                    'is_dative': 'dat' in verb_info['cases'],
                    'is_instrumental': 'ins' in verb_info['cases']
                })
                if 'acc' in verb_info['cases']:
                    context['sentence_structure']['has_direct_object'] = True
                if 'dat' in verb_info['cases']:
                    context['sentence_structure']['has_indirect_object'] = True
        
        # Detect subject based on word order and question patterns
        if context['is_question']:
            context['sentence_structure']['has_subject'] = True
        elif words and words[0] not in prepositions and words[0] not in verb_cases:
            context['sentence_structure']['has_subject'] = True
        
        return context
    
    def get_enhanced_prompts(sentence: str, context: dict, case: str) -> list[tuple[str, float]]:
        """Generate enhanced prompts with improved grammatical pattern recognition"""
        prompts = []
        words = sentence.lower().split()
        params = case_params[case]
        
        # Base prompt with standard weight
        prompts.append((f"{sentence} {tokenizer.mask_token}", 1.0))
        
        # Case-specific prompt strategies with enhanced patterns
        if case == 'nom':
            if context['is_question']:
                # Enhanced subject prompts for questions
                prompts.append((f"{tokenizer.mask_token} {sentence}", params['prompt_weight']))
                prompts.append((f"Кто {tokenizer.mask_token} {sentence}", params['prompt_weight']))
                prompts.append((f"Что {tokenizer.mask_token} {sentence}", params['prompt_weight']))
            elif context['sentence_structure']['has_subject']:
                # Subject prompts for declarative sentences
                prompts.append((f"{tokenizer.mask_token} {sentence}", params['prompt_weight']))
        
        elif case == 'acc':
            # Enhanced direct object prompts
            for verb in context['verbs']:
                if verb['is_transitive']:
                    # Boost prompts with transitive verbs
                    prompts.append((f"{verb['word']} {tokenizer.mask_token}", params['prompt_weight']))
                    prompts.append((f"{sentence} {verb['word']} {tokenizer.mask_token}", params['prompt_weight']))
        
        elif case == 'dat':
            # Enhanced indirect object prompts
            for verb in context['verbs']:
                if verb['is_dative']:
                    # Boost prompts with dative verbs
                    prompts.append((f"{verb['word']} {tokenizer.mask_token}", params['prompt_weight']))
                    prompts.append((f"{sentence} {verb['word']} {tokenizer.mask_token}", params['prompt_weight']))
        
        elif case == 'gen':
            # Enhanced genitive prompts
            for prep in context['prepositions']:
                if prep['is_genitive']:
                    # Boost prompts with genitive prepositions
                    prompts.append((f"{prep['word']} {tokenizer.mask_token}", params['prompt_weight']))
                    prompts.append((f"{sentence} {prep['word']} {tokenizer.mask_token}", params['prompt_weight']))
        
        elif case == 'ins':
            # Enhanced instrumental prompts
            for verb in context['verbs']:
                if verb['is_instrumental']:
                    prompts.append((f"{verb['word']} {tokenizer.mask_token}", params['prompt_weight']))
                    prompts.append((f"{sentence} {verb['word']} {tokenizer.mask_token}", params['prompt_weight']))
        
        elif case == 'prep':
            # Enhanced prepositional prompts
            for prep in context['prepositions']:
                if prep['is_prepositional']:
                    prompts.append((f"{prep['word']} {tokenizer.mask_token}", params['prompt_weight']))
                    prompts.append((f"{sentence} {prep['word']} {tokenizer.mask_token}", params['prompt_weight']))
        
        # Add negation-aware prompts with enhanced weights
        if context['has_negation']:
            prompts.append((f"не {tokenizer.mask_token} {sentence}", params['prompt_weight']))
            prompts.append((f"{sentence} не {tokenizer.mask_token}", params['prompt_weight']))
        
        return list(set(prompts))  # Remove duplicates while preserving weights
    
    def calculate_enhanced_score(logits: torch.Tensor, token_ids: list[int], 
                               temperature: float, context: dict, case: str) -> float:
        """Calculate score with improved grammatical pattern recognition"""
        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Calculate base score for the pronoun tokens
        token_scores = []
        for token_id in token_ids:
            token_scores.append(probs[0, token_id].item())
        
        if not token_scores:
            return 0.0
        
        # Use geometric mean for multi-token pronouns
        base_score = np.exp(np.mean(np.log(np.array(token_scores) + 1e-10)))
        params = case_params[case]
        
        # Apply enhanced case-specific adjustments
        if case == 'nom':
            if context['is_question'] or context['sentence_structure']['has_subject']:
                base_score *= params['context_boost']
        
        elif case == 'acc':
            if context['sentence_structure']['has_direct_object']:
                base_score *= params['context_boost']
            if any(v['is_transitive'] for v in context['verbs']):
                base_score *= 1.2
        
        elif case == 'dat':
            if context['sentence_structure']['has_indirect_object']:
                base_score *= params['context_boost']
            if any(v['is_dative'] for v in context['verbs']):
                base_score *= 1.2
        
        elif case == 'gen':
            if any(p['is_genitive'] for p in context['prepositions']):
                base_score *= params['context_boost']
        
        elif case == 'ins':
            if any(v['is_instrumental'] for v in context['verbs']):
                base_score *= params['context_boost']
        
        elif case == 'prep':
            if context['sentence_structure']['has_prepositional_phrase']:
                base_score *= params['context_boost']
            if any(p['is_prepositional'] for p in context['prepositions']):
                base_score *= 1.2
        
        # Apply negation adjustment
        if context['has_negation']:
            if case in ['gen', 'acc']:  # Common cases with negation
                base_score *= 1.1
        
        return float(base_score)
    
    # Analyze context with enhanced pattern recognition
    context = analyze_context(sentence)
    
    # Calculate scores for each case with improved weighting
    case_scores = {}
    for case_name, case_pronoun in possible_cases.items():
        try:
            # Get case parameters
            params = case_params.get(case_name, {'temperature': 0.8, 'weight': 1.0, 'prompt_weight': 1.0, 'context_boost': 1.0})
            
            # Get enhanced prompts with weights
            prompts_with_weights = get_enhanced_prompts(sentence, context, case_name)
            
            # Get token IDs for the pronoun
            pronoun_token_ids = tokenizer.encode(case_pronoun, add_special_tokens=False)
            
            # Calculate scores for each prompt with improved weighting
            weighted_scores = []
            for prompt, weight in prompts_with_weights:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
                
                if len(mask_token_index) == 0:
                    continue
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    mask_token_logits = outputs.logits[0, mask_token_index, :]
                    
                    # Calculate enhanced score with context
                    prompt_score = calculate_enhanced_score(
                        mask_token_logits,
                        pronoun_token_ids,
                        params['temperature'],
                        context,
                        case_name
                    )
                    weighted_scores.append(prompt_score * weight)
            
            # Use weighted average of prompt scores with improved weighting
            if weighted_scores:
                # Apply case-specific weight and context boost
                case_scores[case_name] = float(np.mean(weighted_scores)) * params['weight']
            else:
                case_scores[case_name] = 0.0
            
        except Exception as e:
            print(f"Error processing case {case_name} for sentence: {sentence}")
            print(f"Error: {str(e)}")
            case_scores[case_name] = 0.0
    
    if not case_scores:
        raise ValueError("No valid cases could be processed")
    
    # Normalize scores to sum to 1 with improved stability
    total_score = sum(case_scores.values())
    if total_score > 0:
        case_scores = {case: score/total_score for case, score in case_scores.items()}
    
    # Return both the predicted case and the scores
    predicted_case = max(case_scores.items(), key=lambda x: x[1])[0]
    return predicted_case, case_scores

def evaluate_case_prediction(model, tokenizer, rumi_data: pd.DataFrame) -> dict:
    """
    Enhanced evaluation of the model's ability to predict the correct case for pronouns.
    Includes additional metrics and analysis.
    """
    # Dictionary to map case names to their column names in the dataset
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
    
    # Add confidence tracking
    confidence_scores = []
    correct_confidences = []
    incorrect_confidences = []
    
    print("\nEvaluating case prediction accuracy...")
    
    for idx, row in rumi_data.iterrows():
        try:
            sentence = row['sentence '].strip()
            correct_case = row['correct_case '].strip()
            
            if pd.isna(sentence) or pd.isna(correct_case):
                continue
            
            # Handle starred cases
            is_starred = '*' in correct_case
            base_case = correct_case.replace('*', '')
            
            # Get all possible pronouns for this case
            possible_cases = {}
            for case, columns in case_columns.items():
                for col in columns:
                    pronoun = row[col]
                    if pd.notna(pronoun) and isinstance(pronoun, str) and pronoun.strip():
                        possible_cases[case] = pronoun.strip()
            
            if not possible_cases:
                continue
            
            # Get model's prediction and scores
            predicted_case, case_scores = predict_case(model, tokenizer, sentence, "", possible_cases)
            
            # Calculate confidence (normalized score)
            total_score = sum(case_scores.values())
            confidence = case_scores[predicted_case] / total_score if total_score > 0 else 0
            confidence_scores.append(confidence)
            
            # Update statistics
            total_predictions += 1
            case_accuracy[base_case]['total'] += 1
            
            # For starred cases, count as correct if either the exact case or base case is predicted
            is_correct = (predicted_case == correct_case) or (is_starred and predicted_case == base_case)
            
            if is_correct:
                correct_predictions += 1
                case_accuracy[base_case]['correct'] += 1
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
            
            # Track prediction patterns
            prediction_patterns[base_case][predicted_case] += 1
            
            # Print progress with confidence information
            if total_predictions % 10 == 0:
                avg_confidence = np.mean(confidence_scores[-10:]) if confidence_scores else 0
                print(f"Processed {total_predictions} examples...")
                print(f"Current accuracy: {correct_predictions/total_predictions:.2%}")
                print(f"Average confidence: {avg_confidence:.2%}")
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate per-case accuracy and confidence metrics
    case_results = {}
    for case, stats in case_accuracy.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        case_results[case] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total'],
            'avg_confidence': np.mean([c for c, p in zip(confidence_scores, prediction_patterns[case].values()) if p > 0]) if any(prediction_patterns[case].values()) else 0
        }
    
    # Calculate confidence metrics
    confidence_metrics = {
        'overall_avg_confidence': np.mean(confidence_scores),
        'correct_avg_confidence': np.mean(correct_confidences) if correct_confidences else 0,
        'incorrect_avg_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
        'confidence_correlation': np.corrcoef(confidence_scores, [1 if c in correct_confidences else 0 for c in confidence_scores])[0,1] if len(confidence_scores) > 1 else 0
    }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'case_results': case_results,
        'prediction_patterns': prediction_patterns,
        'confidence_metrics': confidence_metrics
    }

# Load model and data
print("Loading model and data...")
model_name = "xlm-roberta-base"  # Changed to XLM-RoBERTa
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    low_cpu_mem_usage=True,
    device_map="auto",  # Added for better memory management
    attn_implementation="eager"  # Fix attention implementation warning
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load the Russian pronoun dataset
rumi_data = pd.read_csv('rumi.csv', encoding='utf-8')

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
        print(f"  Attention to mask: {head_analysis['attention_to_mask']:.3f}")
        print(f"  Attention from mask: {head_analysis['attention_from_mask']:.3f}")

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

# Print to console
print("\nEvaluation Results:")
print(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.2%}")
print(f"Total Predictions: {evaluation_results['total_predictions']}")
print(f"Correct Predictions: {evaluation_results['correct_predictions']}")
print("\nPer-Case Accuracy:")
for case, results in evaluation_results['case_results'].items():
    if results['total'] > 0:  # Only show cases that appear in the data
        print(f"{case}: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

# Save results to file
with open('xlmroberta_accuracy.txt', 'w', encoding='utf-8') as f:
    f.write("XLM-RoBERTa Case Prediction Evaluation Results\n")
    f.write("============================================\n\n")
    f.write(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.2%}\n")
    f.write(f"Total Predictions: {evaluation_results['total_predictions']}\n")
    f.write(f"Correct Predictions: {evaluation_results['correct_predictions']}\n\n")
    
    f.write("Confidence Metrics:\n")
    f.write(f"Overall Average Confidence: {evaluation_results['confidence_metrics']['overall_avg_confidence']:.2%}\n")
    f.write(f"Average Confidence for Correct Predictions: {evaluation_results['confidence_metrics']['correct_avg_confidence']:.2%}\n")
    f.write(f"Average Confidence for Incorrect Predictions: {evaluation_results['confidence_metrics']['incorrect_avg_confidence']:.2%}\n")
    f.write(f"Confidence-Accuracy Correlation: {evaluation_results['confidence_metrics']['confidence_correlation']:.3f}\n\n")
    
    f.write("Per-Case Accuracy and Confidence:\n")
    for case, results in evaluation_results['case_results'].items():
        if results['total'] > 0:
            f.write(f"{case}:\n")
            f.write(f"  Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})\n")
            f.write(f"  Average Confidence: {results['avg_confidence']:.2%}\n")
    
    f.write("\nPrediction Patterns Analysis:\n")
    for true_case, predictions in evaluation_results['prediction_patterns'].items():
        if evaluation_results['case_results'][true_case]['total'] > 0:
            f.write(f"\nWhen correct case is {true_case}:\n")
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            for pred_case, count in sorted_predictions:
                if count > 0:
                    percentage = count / evaluation_results['case_results'][true_case]['total'] * 100
                    f.write(f"  Predicted as {pred_case}: {count} times ({percentage:.1f}%)\n")
