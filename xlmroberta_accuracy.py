import torch
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import sys
import io

# Set stdout to handle UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
    Enhanced prediction of which case should be used for a pronoun in a given sentence using masked language modeling.
    Implements multiple strategies to improve accuracy:
    1. Case-specific context windows and scoring
    2. Preposition and verb agreement analysis
    3. Ensemble of different prediction strategies
    4. Improved context analysis with word order
    5. Case-specific boosting based on grammatical patterns
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Case-specific parameters with refined values based on results
    case_params = {
        'nom': {
            'temperature': 0.4,  # Lower temperature for more confident predictions
            'context_weight': 1.3,
            'position_weight': 1.2,  # Nominative often appears at start
            'verb_boost': 1.4  # Strong boost for verb agreement
        },
        'dat': {
            'temperature': 0.7,
            'context_weight': 1.6,  # Increased for better preposition handling
            'prep_boost': 1.5,  # Strong boost for preposition agreement
            'verb_boost': 1.3
        },
        'gen': {
            'temperature': 0.5,
            'context_weight': 1.2,
            'prep_boost': 1.4,
            'verb_boost': 1.3,
            'negation_boost': 1.5  # Genitive often follows negation
        },
        'acc': {
            'temperature': 0.6,
            'context_weight': 1.2,
            'verb_boost': 1.5,  # Strong boost for verb agreement
            'position_weight': 1.1  # Often follows verb
        },
        'ins': {
            'temperature': 0.6,
            'context_weight': 1.4,
            'verb_boost': 1.4,
            'prep_boost': 1.3
        },
        'prep': {
            'temperature': 0.5,
            'context_weight': 1.5,
            'prep_boost': 1.6,  # Strongest boost for preposition agreement
            'position_weight': 1.2  # Often follows preposition
        }
    }
    
    # Expanded preposition dictionary with more context
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
    
    # Expanded verb dictionary with more context and case patterns
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
    
    def analyze_sentence_context(sentence: str) -> dict:
        """Enhanced sentence context analysis with word order and grammatical patterns"""
        words = sentence.lower().split()
        context_info = {
            'prepositions': [],
            'verbs': [],
            'preceding_word': words[-2] if len(words) > 1 else None,
            'following_word': words[1] if len(words) > 1 else None,
            'word_order': [],
            'negation': False,
            'question': False
        }
        
        # Check for negation and questions
        context_info['negation'] = any(word in ['не', 'нет', 'ни'] for word in words)
        context_info['question'] = any(word in ['что', 'кто', 'где', 'когда', 'почему', 'как'] for word in words)
        
        # Analyze word order and relationships
        for i, word in enumerate(words):
            # Find prepositions and their context
            if word in prepositions:
                prep_info = prepositions[word]
                context_info['prepositions'].append({
                    'word': word,
                    'cases': prep_info['cases'],
                    'context': prep_info['context'],
                    'position': i,
                    'next_word': words[i + 1] if i + 1 < len(words) else None
                })
            
            # Find verbs and their context
            if word in verb_cases:
                verb_info = verb_cases[word]
                context_info['verbs'].append({
                    'word': word,
                    'cases': verb_info['cases'],
                    'context': verb_info['context'],
                    'position': i,
                    'next_word': words[i + 1] if i + 1 < len(words) else None
                })
            
            # Track word order for case-specific analysis
            context_info['word_order'].append({
                'word': word,
                'position': i,
                'is_preposition': word in prepositions,
                'is_verb': word in verb_cases
            })
        
        return context_info
    
    def get_case_specific_prompts(sentence: str, case: str) -> list[str]:
        """Generate enhanced case-specific prompts based on detailed context analysis"""
        context = analyze_sentence_context(sentence)
        prompts = []
        
        # Base prompts with position awareness
        if case == 'nom':
            # Nominative often appears at start or after certain verbs
            prompts.extend([
                f"{tokenizer.mask_token} {sentence}",  # Start position
                f"{sentence} {tokenizer.mask_token}"   # End position
            ])
        else:
            prompts.extend([
                f"{sentence} {tokenizer.mask_token}",  # Original
                f"{tokenizer.mask_token} {sentence}",  # Reversed
                f"{sentence[:len(sentence)//2]} {tokenizer.mask_token} {sentence[len(sentence)//2:]}"  # Split context
            ])
        
        # Add preposition-aware prompts
        if context['prepositions']:
            for prep_info in context['prepositions']:
                if case in prep_info['cases']:
                    # Create prompts that emphasize preposition context
                    prompts.extend([
                        f"{prep_info['word']} {tokenizer.mask_token} {sentence.replace(prep_info['word'], '').strip()}",
                        f"{sentence} {prep_info['word']} {tokenizer.mask_token}",
                        f"{prep_info['word']} {tokenizer.mask_token} {context['following_word'] if context['following_word'] else ''}"
                    ])
        
        # Add verb-aware prompts
        if context['verbs']:
            for verb_info in context['verbs']:
                if case in verb_info['cases']:
                    # Create prompts that emphasize verb context
                    prompts.extend([
                        f"{verb_info['word']} {tokenizer.mask_token} {sentence.replace(verb_info['word'], '').strip()}",
                        f"{sentence} {verb_info['word']} {tokenizer.mask_token}",
                        f"{verb_info['word']} {tokenizer.mask_token} {context['following_word'] if context['following_word'] else ''}"
                    ])
        
        # Add negation-aware prompts for genitive
        if case == 'gen' and context['negation']:
            prompts.extend([
                f"не {tokenizer.mask_token} {sentence.replace('не', '').strip()}",
                f"{sentence} не {tokenizer.mask_token}"
            ])
        
        # Add question-aware prompts
        if context['question']:
            prompts.extend([
                f"{tokenizer.mask_token} ? {sentence}",
                f"{sentence} {tokenizer.mask_token} ?"
            ])
        
        return list(set(prompts))  # Remove duplicates
    
    # Calculate scores for each case using ensemble approach
    case_scores = {}
    for case_name, case_pronoun in possible_cases.items():
        try:
            # Get case-specific parameters
            params = case_params.get(case_name, {'temperature': 0.7, 'context_weight': 1.0})
            temperature = params['temperature']
            context_weight = params['context_weight']
            
            # Get case-specific prompts
            prompts = get_case_specific_prompts(sentence, case_name)
            
            # Get all token IDs for the pronoun
            pronoun_token_ids = tokenizer.encode(case_pronoun, add_special_tokens=False)
            
            # Calculate scores for each prompt
            prompt_scores = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
                
                if len(mask_token_index) == 0:
                    continue
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    mask_token_logits = outputs.logits[0, mask_token_index, :]
                    
                    # Apply temperature scaling
                    scaled_logits = mask_token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Calculate score considering all tokens in the pronoun
                    token_scores = []
                    for i, token_id in enumerate(pronoun_token_ids):
                        if i == 0:
                            token_scores.append(probs[0, token_id].item())
                        else:
                            prev_tokens = tokenizer.decode(pronoun_token_ids[:i])
                            new_prompt = f"{sentence} {prev_tokens} {tokenizer.mask_token}"
                            new_inputs = tokenizer(new_prompt, return_tensors="pt").to(device)
                            new_mask_index = torch.where(new_inputs["input_ids"] == tokenizer.mask_token_id)[1]
                            
                            if len(new_mask_index) > 0:
                                new_outputs = model(**new_inputs)
                                new_logits = new_outputs.logits[0, new_mask_index, :] / temperature
                                new_probs = torch.softmax(new_logits, dim=-1)
                                token_scores.append(new_probs[0, token_id].item())
                    
                    if token_scores:
                        # Use log-sum-exp trick for numerical stability
                        log_scores = np.log(np.array(token_scores) + 1e-10)
                        prompt_score = np.exp(np.mean(log_scores))
                        
                        # Apply context-specific weighting
                        context = analyze_sentence_context(sentence)
                        
                        # Apply case-specific boosts
                        if case_name == 'nom' and any(w['is_verb'] for w in context['word_order']):
                            prompt_score *= params.get('verb_boost', 1.0)
                        elif case_name in ['dat', 'prep'] and any(w['is_preposition'] for w in context['word_order']):
                            prompt_score *= params.get('prep_boost', 1.0)
                        elif case_name == 'gen' and context['negation']:
                            prompt_score *= params.get('negation_boost', 1.0)
                        elif case_name in ['acc', 'ins'] and any(w['is_verb'] for w in context['word_order']):
                            prompt_score *= params.get('verb_boost', 1.0)
                        
                        # Apply position-based weighting
                        if case_name in ['nom', 'acc'] and context['word_order']:
                            position = context['word_order'][0]['position']
                            if (case_name == 'nom' and position == 0) or \
                               (case_name == 'acc' and position > 0):
                                prompt_score *= params.get('position_weight', 1.0)
                        
                        prompt_scores.append(prompt_score)
            
            # Average score across different prompts with weighting
            if prompt_scores:
                # Weight later prompts more heavily as they're more specific
                weights = np.linspace(1.0, 1.5, len(prompt_scores))
                case_scores[case_name] = float(np.average(prompt_scores, weights=weights))
            else:
                case_scores[case_name] = 0.0
            
        except Exception as e:
            print(f"Error processing case {case_name} for sentence: {sentence}")
            print(f"Error: {str(e)}")
            case_scores[case_name] = 0.0
    
    if not case_scores:
        raise ValueError("No valid cases could be processed")
    
    # Apply final context-based boosting
    context = analyze_sentence_context(sentence)
    
    # Boost scores based on grammatical patterns
    for case_name in case_scores:
        # Preposition-based boosting
        if context['prepositions']:
            for prep_info in context['prepositions']:
                if case_name in prep_info['cases']:
                    case_scores[case_name] *= params.get('prep_boost', 1.2)
        
        # Verb-based boosting
        if context['verbs']:
            for verb_info in context['verbs']:
                if case_name in verb_info['cases']:
                    case_scores[case_name] *= params.get('verb_boost', 1.2)
        
        # Special case handling
        if case_name == 'gen' and context['negation']:
            case_scores[case_name] *= params.get('negation_boost', 1.5)
        elif case_name == 'nom' and not any(w['is_preposition'] for w in context['word_order']):
            case_scores[case_name] *= params.get('position_weight', 1.2)
    
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
    device_map="auto"  # Added for better memory management
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
