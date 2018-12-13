import pandas as pd

# Read inputs
stances_df = pd.read_csv("../data/train_stances.csv")
bodies_df = pd.read_csv("../data/train_bodies.csv")

# Match headlines with bodies
bodies_df.set_index(['Body ID'], inplace = True)
stances_df["body"] = stances_df["Body ID"].apply(lambda r: bodies_df.loc[r]['articleBody'])

# Strip columns of commas
output_df = stances_df[["Headline", "body", "Stance"]].copy()
for col in ['Headline', 'body']:
    for skip_var in [',', '"', '\n']:
        output_df[col] = output_df[col].apply(lambda h: h.replace(skip_var, ''))

# Map tag to int
tags = {'unrelated': 0, 'agree': 1, 'disagree': 2, 'discuss': 3}
output_df['Stance'] = output_df['Stance'].apply(lambda r: tags[r])

# Save to disk
output_df.to_csv("../data/bert_format/train.csv", header = False, index = False)
