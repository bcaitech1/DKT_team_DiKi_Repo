import pandas as pd
import numpy as np


def main():
    ensemble_list = ['./output/output_1.csv', './output/output_2.csv', './output/output_4.csv', './output/output_5.csv']

    pred_list = []
    for path in ensemble_list:
        df = pd.read_csv(path)

        pred_list.append(df['prediction'].to_numpy())

    pred_list = np.array(pred_list)
    
    ensembled_pred = np.mean(pred_list, axis=0)

    ensembled_df = pd.DataFrame(ensembled_pred, columns=['prediction'])
    ensembled_df.to_csv('./output/ensembled.csv', index_label='id')
    


if __name__ == '__main__':
    main()