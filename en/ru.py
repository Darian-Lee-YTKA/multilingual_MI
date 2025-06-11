import pandas as pd
## load the russian_mi_dataset.csv
df = pd.read_csv('russian_mi_dataset.csv')

## print the columns
print(df.columns)

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# filter out rows with correct_case = "ins"
# test_cases = ["dat","gen","acc","ins"]
df_filtered = df[df['correct_case'] == "ins"]

## Use Bloom to predict the correct case for the first 10 rows
from transformers import BloomForCausalLM, BloomTokenizerFast

model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-3b")

