import numpy as np
import pandas as pd
from collections import Counter
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix

def main():
    data_df = pd.read_csv('./data/train.csv')

    defects_df = []
    for i in range(0, len(data_df), 4):
        defi = {}
        defi['ImageId_ClassId'] = data_df.loc[i, 'ImageId_ClassId'][:-2]
        defi['1'] = int(not pd.isnull(data_df.loc[i, 'EncodedPixels']))
        defi['2'] = int(not pd.isnull(data_df.loc[i+1, 'EncodedPixels']))
        defi['3'] = int(not pd.isnull(data_df.loc[i+2, 'EncodedPixels']))
        defi['4'] = int(not pd.isnull(data_df.loc[i+3, 'EncodedPixels']))
        defects_df.append(defi)
    defects_df = pd.DataFrame(defects_df)[['ImageId_ClassId', '1', '2', '3', '4']]
    # defects_df.to_csv('./data/defect_types.csv', index=False)

    Xd = np.expand_dims(np.array(range(len(defects_df))), 1)
    X_train, y_train, X_test, y_test = iterative_train_test_split(Xd, defects_df[['1', '2', '3', '4']].to_numpy(), test_size = 0.2)

    print("Train set size = {}, Test set size = {}".format(len(X_train), len(X_test)))

    print(pd.DataFrame({
        'train': Counter(str(combination) for row in get_combination_wise_output_matrix(y_train, order=2) for combination in row),
        'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for combination in row)
    }).T.fillna(0.0))

    defects_df['is_valid'] = False
    defects_df.loc[X_test[:,0], 'is_valid'] = True
    defects_df.to_csv('./data/split.csv', index=False)

if __name__ == '__main__':
    main()
