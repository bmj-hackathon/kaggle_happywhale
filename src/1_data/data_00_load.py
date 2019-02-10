#%%

PATH_INPUT = Path.cwd() / 'input'
assert PATH_INPUT.exists()

# %%
train_df = pd.read_csv(PATH_INPUT / 'train.csv')
train_df.head()

# %%

train_df.sample(10)

