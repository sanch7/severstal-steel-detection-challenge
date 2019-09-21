from fastai.vision import *

def plot_one_hot_ds(ds, idx=0, def_type=1):
    for i, bt in enumerate(iter(ds)):
        if i == idx:
            break
    img = bt[0].data.numpy()
    plt.imshow(np.rollaxis(img, 0, 3))
    plt.imshow(bt[1].data[def_type-1], alpha=0.2)

class ImageSegmentFloat(ImageSegment):
    def __init__(self, px:Tensor):
        "Create from raw tensor image data `px`."
        self._px = px.type(torch.FloatTensor)
        self._logit_px=None
        self._flow=None
        self._affine_mat=None
        self.sample_kwargs = {}

    @property
    def data(self)->TensorImage:
        "Return this image pixels as a `FloatTensor`."
        return self.px.float()

class SegmentationLabelListOneHot(SegmentationLabelList):
    def open(self, fn):
        mask = open_mask(fn).data
        mask = torch.cat([mask==1, mask==2, mask==3, mask==4], dim=0).float()
        return ImageSegmentFloat(mask)

class SegmentationItemListOneHot(SegmentationItemList):
    _label_cls = SegmentationLabelListOneHot

def get_steel_transforms(config):
    train_tfms = [
                  # # crop_pad only center cropping for some reason
                  # RandTransform(tfm=TfmCrop (crop_pad), 
                  #       kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, 
                  #       p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmPixel (crop), 
                    kwargs={'size': config.imsize, 'row_pct': (0, 1), 'col_pct': (0, 1)}, 
                    p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmPixel (flip_lr), 
                        kwargs={}, 
                        p=0.5, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmCoord (symmetric_warp), 
                    kwargs={'magnitude': (-0.2, 0.2)}, 
                    p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmAffine (rotate), 
                    kwargs={'degrees': (-10.0, 10.0)}, 
                    p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmAffine (zoom), 
                    kwargs={'scale': (1.0, 1.1), 'row_pct': (0, 1), 'col_pct': (0, 1)}, 
                    p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmLighting (brightness), 
                    kwargs={'change': (0.4, 0.6)}, 
                    p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmLighting (contrast), 
                    kwargs={'scale': (0.8, 1.25)}, 
                    p=0.75, resolved={}, do_run=True, is_random=True, use_on_y=True)
                  ]

    valid_tfms = [
                  # # crop_pad only center cropping for some reason
                  # RandTransform(tfm=TfmCrop (crop_pad), 
                  #   kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, 
                  #   p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
                  RandTransform(tfm=TfmPixel (crop), 
                    kwargs={'size': config.imsize, 'row_pct': (0, 1), 'col_pct': (0, 1)}, 
                    p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
                 ]
    return (train_tfms, valid_tfms)


def get_label_dict(val_df):
    label_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
    for i in range(len(val_df)):
        labels = []
        if val_df.loc[i, '1']!=0: labels.append(1)
        if val_df.loc[i, '2']!=0: labels.append(2)
        if val_df.loc[i, '3']!=0: labels.append(3)
        if val_df.loc[i, '4']!=0: labels.append(4)
        if len(labels) == 0: labels.append(0)
        for label in labels:
            label_dict[label].append(i)
    return label_dict


def get_label_freq(val_df):
    labelfreq = [((val_df['1'] + val_df['2'] + val_df['3'] + val_df['4']) == 0).sum()]
    labelfreq.extend([(val_df[i]!=0).sum() for i in ['1', '2', '3', '4']])
    labelfreq = np.array(labelfreq)
    return labelfreq


def oversample_train(split_df, random_seed=42):
    """Oversamples so that all classes (including background) have approximately the same freq"""
    random.seed(random_seed)
    train_df = split_df[split_df['is_valid'] == False].reset_index(drop=True)
    val_df = split_df[split_df['is_valid'] == True].reset_index(drop=True)

    label_dict = get_label_dict(train_df)
    labelfreq = get_label_freq(train_df)
    diff = labelfreq.max()-labelfreq.min()
    while (diff > 50):
        dup = random.choices(label_dict[labelfreq.argmin()], k=diff)
        train_df = train_df.append(train_df.loc[dup], ignore_index=True)
        labelfreq = get_label_freq(train_df)
        diff = labelfreq.max()-labelfreq.min()

    split_df = train_df.append(val_df, ignore_index=True)
    return split_df


def get_data_bunch(split_df, config):
    train_data_paths, valid_data_paths, train_label_paths, valid_label_paths = [], [], [], []

    if config.oversample:
        split_df = oversample_train(split_df)

    for i in range(len(split_df)):
        data_path = './data/train_images/' + split_df.loc[i, 'ImageId_ClassId']
        label_path = './data/train_masks/' + split_df.loc[i, 'ImageId_ClassId'].replace('.jpg', '.png')
        if split_df.loc[i, 'is_valid']:
            if config.load_valid_crops:
                for i in range(7):
                    valid_data_paths.append(Path(data_path.replace('.jpg', '_c{}.jpg'.format(i))))
                    valid_label_paths.append(Path(label_path.replace('.png', '_c{}.png'.format(i))))
            else:
                valid_data_paths.append(Path(data_path))
                valid_label_paths.append(Path(label_path))
        else:
            if config.load_valid_crops:
                for i in range(config.train_duplicate):      # So we don't spend a lot of time validating
                    if config.load_train_crops:
                        train_data_paths.append(Path(data_path.replace('.jpg', '_c{}.jpg'.format(i))))
                        train_label_paths.append(Path(label_path.replace('.png', '_c{}.png'.format(i))))
                    else:
                        train_data_paths.append(Path(data_path))
                        train_label_paths.append(Path(label_path))
            else:
                train_data_paths.append(Path(data_path))
                train_label_paths.append(Path(label_path))

    seg_item_list = SegmentationItemListOneHot if config.one_hot_labels else SegmentationItemList 
    train = seg_item_list(train_data_paths)
    valid = seg_item_list(valid_data_paths)
    # train_label = SegmentationLabelList(train_label_paths)
    # valid_label = SegmentationLabelList(valid_label_paths)
    src = (seg_item_list.from_folder('.')
            .split_by_list(train=train, valid=valid)
            .label_from_lists(train_labels=train_label_paths, valid_labels=valid_label_paths, classes=[1, 2, 3, 4]))

    data = (src.transform(get_steel_transforms(config=config), size=config.imsize, tfm_y=True)
            .databunch(bs=config.batch_size)
            .normalize(imagenet_stats))

    return data
