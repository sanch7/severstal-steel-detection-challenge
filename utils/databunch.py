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

def get_steel_transforms(size=256):
    train_tfms = [
                  # # crop_pad only center cropping for some reason
                  # RandTransform(tfm=TfmCrop (crop_pad), 
                  #       kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, 
                  #       p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True),
                  RandTransform(tfm=TfmPixel (crop), 
                    kwargs={'size': size, 'row_pct': (0, 1), 'col_pct': (0, 1)}, 
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
                    kwargs={'size': size, 'row_pct': (0, 1), 'col_pct': (0, 1)}, 
                    p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
                 ]
    return (train_tfms, valid_tfms)


def get_data_bunch(split_df, size=256, batch_size=1, one_hot=True, load_valid_crops=True, load_train_crops=False):
    train_data_paths, valid_data_paths, train_label_paths, valid_label_paths = [], [], [], []
    for i in range(len(split_df)):
        data_path = './data/train_images/' + split_df.loc[i, 'ImageId_ClassId']
        label_path = './data/train_masks/' + split_df.loc[i, 'ImageId_ClassId'].replace('.jpg', '.png')
        if split_df.loc[i, 'is_valid']:
            if load_valid_crops:
                for i in range(7):
                    valid_data_paths.append(Path(data_path.replace('.jpg', '_c{}.jpg'.format(i))))
                    valid_label_paths.append(Path(label_path.replace('.png', '_c{}.png'.format(i))))
            else:
                valid_data_paths.append(Path(data_path))
                valid_label_paths.append(Path(label_path))
        else:
            if load_valid_crops:
                for i in range(7):      # So we don't spend a lot of time validating
                    if load_train_crops:
                        train_data_paths.append(Path(data_path.replace('.jpg', '_c{}.jpg'.format(i))))
                        train_label_paths.append(Path(label_path.replace('.png', '_c{}.png'.format(i))))
                    else:
                        train_data_paths.append(Path(data_path))
                        train_label_paths.append(Path(label_path))
            else:
                train_data_paths.append(Path(data_path))
                train_label_paths.append(Path(label_path))

    seg_item_list = SegmentationItemListOneHot if one_hot else SegmentationItemList 
    train = seg_item_list(train_data_paths)
    valid = seg_item_list(valid_data_paths)
    # train_label = SegmentationLabelList(train_label_paths)
    # valid_label = SegmentationLabelList(valid_label_paths)
    src = (seg_item_list.from_folder('.')
            .split_by_list(train=train, valid=valid)
            .label_from_lists(train_labels=train_label_paths, valid_labels=valid_label_paths, classes=[1, 2, 3, 4]))

    data = (src.transform(get_steel_transforms(size=size), size=size, tfm_y=True)
            .databunch(bs=batch_size)
            .normalize(imagenet_stats))

    return data
