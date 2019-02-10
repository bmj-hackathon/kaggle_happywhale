#%%

PATH_INPUT = Path.cwd() / 'input'
assert PATH_INPUT.exists()

# %%
train_df = pd.read_csv(PATH_INPUT / 'train.csv')
# train_df.head()
logging.info("Train {}".format(train_df.shape))
# %%

train_df.sample(10)

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
ROWS = 3
COLS = 3
NUM_IMAGES = ROWS * COLS

sel_img_fnames = [row[1]['Image'] for row in train_df.sample(NUM_IMAGES).iterrows()]
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

for i, ts_string_index in enumerate(ts_string_indices):
    rec = dataset.df.loc[ts_string_index]

    timestamp_string = rec['datetime'].strftime("%D %H:%M:%S.") + "{:.2}".format(
        str(rec['datetime'].microsecond))

    if 'steering_pred_signal' in dataset.df.columns:
        this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,
                                                                              rec['steering_signal'],
                                                                              rec['steering_pred_signal'],
                                                                              rec['throttle_signal'])
    else:
        this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string, rec['steering_signal'],
                                                                       rec['throttle_signal'])

    ax = fig.add_subplot(ROWS, COLS, i + 1)

    # Main Image ##########################################################
    jpg_path = os.path.join(dataset.path_dataset, source_jpg_folder, ts_string_index + '.' + extension)
    assert os.path.exists(jpg_path), "{} does not exist".format(jpg_path)
    img = mpl.image.imread(jpg_path)
    ax.imshow(img, cmap=cmap)
    # plt.title(str_label)

    # Steering widget HUD #################################################
    # Steering HUD: Actual steering signal
    steer_actual = ''.join(['|' if v else '-' for v in dataset.linear_bin(rec['steering_signal'])])
    text_steer = ax.text(80, 105, steer_actual, fontdict=font_steering, horizontalalignment='center',
                         verticalalignment='center', color=gui_color)
    # Steering HUD: Predicted steering angle
    if 'steering_pred_signal' in dataset.df.columns:
        steer_pred = ''.join(['◈' if v else ' ' for v in dataset.linear_bin(rec['steering_pred_signal'])])
        text_steer_pred = ax.text(80, 95, steer_pred, fontdict=font_steering, horizontalalignment='center',
                                  verticalalignment='center', color='red')

outpath = os.path.join(dataset.path_dataset, outfname)
fig.savefig(outpath)
logging.debug("Wrote Sample Frames figure to {}".format(outpath))

