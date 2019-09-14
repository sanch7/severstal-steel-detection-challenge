from fastai.vision import *

def get_steel_transforms():
    train_tfms = [
                  RandTransform(tfm=TfmCrop (crop_pad), 
                        kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, 
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
                  RandTransform(tfm=TfmCrop (crop_pad), 
                    kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, 
                    p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)
                 ]
    return (train_tfms, valid_tfms)


def get_data_bunch(split_df):
    train_data_paths, valid_data_paths, train_label_paths, valid_label_paths = [], [], [], []
    for i in range(len(split_df)):
        data_path = './data/train_images/' + split_df.loc[i, 'ImageId_ClassId']
        label_path = './data/train_masks/' + split_df.loc[i, 'ImageId_ClassId'].replace('.jpg', '.png')
        if split_df.loc[i, 'is_valid']:
            for i in range(7):
                valid_data_paths.append(Path(data_path.replace('.jpg', '_c{}.jpg'.format(i))))
                valid_label_paths.append(Path(label_path.replace('.png', '_c{}.png'.format(i))))
        else:
            for _ in range(7):      # So we don't spend a lot of time validating
                train_data_paths.append(Path(data_path))
                train_label_paths.append(Path(label_path))

    train = SegmentationItemList(train_data_paths)
    valid = SegmentationItemList(valid_data_paths)
    # train_label = SegmentationLabelList(train_label_paths)
    # valid_label = SegmentationLabelList(valid_label_paths)
    src = (SegmentationItemList.from_folder('.')
            .split_by_list(train=train, valid=valid)
            .label_from_lists(train_labels=train_label_paths, valid_labels=valid_label_paths, classes=[1, 2, 3, 4]))

    data = (src.transform(get_steel_transforms(), size=256, tfm_y=True)
            .databunch(bs=9)
            .normalize(imagenet_stats))

    return data

