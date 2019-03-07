#%%

PATH_INPUT = Path.cwd() / 'input'
assert PATH_INPUT.exists()

# %%
df_all = pd.read_csv(PATH_INPUT / 'train.csv')
# train_df.head()
logging.info("Train {}".format(df_all.shape))
# %%

df_all.sample(10)

#%% SETTINGS

font_label_box = {
    'color': 'green',
    'size': 16,
}
font_steering = {'family': 'monospace',
                 # 'color':  'darkred',
                 'weight': 'normal',
                 'size': 20,
                 }

#%%
ROWS = 4
COLS = 3
NUM_IMAGES = ROWS * COLS

sel_img_fnames = [row[1]['Image'] for row in df_all.sample(NUM_IMAGES).iterrows()]
sel_img_paths = [PATH_INPUT / 'train' / name for name in sel_img_fnames]
assert all([p.exists() for p in sel_img_paths])
assert len(sel_img_paths) % 2 == 0

#%%
fig = plt.figure(figsize=PAPER['A3_LANDSCAPE'], facecolor='white')
fig.suptitle("Test {}".format('TEst'), fontsize=20)

for i, img_path in enumerate(sel_img_paths):
    logging.info("{}".format(img_path))
    ax = fig.add_subplot(ROWS, COLS, i + 1)
    img = mpl.image.imread(img_path)
    ax.imshow(img)
    ax.axis('off')
    # plt.title(str_label)
plt.show()
# outpath = os.path.join(dataset.path_dataset, outfname)
# fig.savefig(outpath)
# logging.debug("Wrote Sample Frames figure to {}".format(outpath))

#%%


