

import pandas as pd

import pandas as pd

df = pd.read_csv('russian_mi_dataset.csv')


df.columns = df.columns.str.strip()

# This part of code organizes the different columns (including irregular cases) into classes
# 6 lists for each case
# order in list: masc, fem, [if irregular exists: irreg_masc, irreg_fem, irreg_plural]
gen = ['gen', 'gen_fem', 'gen_plural']

# dat_n (irregular dat masc)= нему (к нему),
# ins_fem (irregular dat fem) = ней (к ней),
# ins (irregular dat plural) = ним (к ним)
dat = ['dat', 'dat_fem', 'dat_plural', 'dat_n', 'ins_fem', 'ins']

acc = ['acc', 'acc_fem', 'acc_plural']
# dat_plural = (irregular ins masc) = им (я горжусь им),
# dat_fem (irregular ins fem) = ей (я горжусь ей),
# ins_plural_no_n (irregular ins plural) = им (я горжусь им
ins = ['ins', 'ins_fem', 'ins_plural', 'dat_plural', 'dat_fem', 'ins_plural_no_n']
nom = ['nom', 'nom_fem', 'nom_plural']
prep = ['prep', 'prep_fem', 'prep_plural']


# just converting from string to the name of a list
cases = {
    'gen': gen,
    'dat': dat,
    'acc': acc,
    'ins': ins,
    'nom': nom,
    'prep': prep,
}


# our output files for pairwise comparisons.
# Each will have column for sentence, column for each incorrect case, and comment column
# column for each case will contain the logit difference between the correct case and this case
# comment column will contain info about when the logit lens was applied (ex. 27 if after layer 27)
output_paths = {
    'acc': 'acc.csv',
    'gen': 'gen.csv',
    'dat': 'dat.csv',
    'ins': 'ins.csv',
}

# setting up the files
for case_key, path in output_paths.items():
    if not os.path.exists(path):
        columns = ['sentence'] + [c for c in cases if c != case_key] + ['comment']
        pd.DataFrame(columns=columns).to_csv(path, index=False)



# most of the next code is just because I wrote the dataset in a stupid way. essentially its going from col name to correct form
form_lookup = {}
for col in df.columns:
    form = str(df[col].iloc[0]).strip()
    if pd.notna(form):
        form_lookup[col] = form



# reverses dict to get form for each token
def get_token_to_column_mapping(form_lookup):
    return {form: col for col, form in form_lookup.items()}

token_to_column = get_token_to_column_mapping(form_lookup)

# Turn LLM logits into logits_dict
def extract_logits_by_token(logits, tokenizer, token_to_column, target_position):
    '''
    :param logits: torch tensor of shape (seq_len, vocab_size)
    :param tokenizer: tokenizer object
    :param token_to_column: dict like {'его': 'acc', 'ей': 'dat_fem'}
    :param target_position: position in sequence to extract from
    :return: dict like {'acc': 1.23, 'dat_fem': 0.98, ...}
    '''
    logits_dict = {}
    for token_str, column_name in token_to_column.items():
        token_ids = tokenizer(token_str, add_special_tokens=False)['input_ids']
        if len(token_ids) != 1:
            continue  # skip compound/multi-token cases
        token_id = token_ids[0]
        try:
            logits_dict[column_name] = logits[target_position, token_id].item()
        except IndexError:
            logits_dict[column_name] = None
    return logits_dict


# This function compares the logits of correct to the logits of incorrect case.
# If the correct case contains possible irregular options, we will first see whether
# irregular or no-irregular options are higher and use the higher one
# then it will take the mean difference between correct and incorrect cases across all genders and write that to the correct file
def pairwise_logit_comparison_across_case(correct, sentence, logits, comment):
    '''
    :param correct: The correct case for that sentence
    :param sentence: The original sentence
    :param logits: All unsorted logits returned from the model
    :param comment: where logit lens was taken
    :return: writes to file
    '''



    correct_cols = cases[correct] # just converting from string to the name of a list


    reg_cols = correct_cols[:3]  # masc, fem, plural
    irreg_cols = correct_cols[3:] if len(correct_cols) > 3 else [] # if there are irregular, store them seperately and compare

    # compare irregular and regular logits to determine which is correct (we assume the one with higher prop is correct)
    reg_logits = [logits.get(col) for col in reg_cols if col in logits]
    irreg_logits = [logits.get(col) for col in irreg_cols if col in logits]

    from statistics import mean
    avg_reg = mean(reg_logits) if reg_logits else float('-inf')
    avg_irreg = mean(irreg_logits) if irreg_logits else float('-inf')
    correct_logit = max(avg_reg, avg_irreg)

    output = {'sentence': sentence}

    # тnow we do pairwise comparison of each other case with the correct case
    for case_name, columns in cases.items():
        if case_name == correct:
            continue
        wrong_cols = columns[:3]
        wrong_logits = [logits.get(col) for col in wrong_cols if col in logits]
        if wrong_logits:
            diff = correct_logit - mean(wrong_logits)
            output[case_name] = round(diff, 8)
        else:
            output[case_name] = 'NA'


    output['comment'] = comment


    print(output)