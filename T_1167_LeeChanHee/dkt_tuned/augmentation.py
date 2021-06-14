import numpy as np

def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.max_seq_len

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 seq_len이 window 크기보다 작으면 augmentation 안한다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride)

            # 앞에서부터 슬라이딩 윈도우 적용
            for window_i in range(total_window):
                window_data = [] # 윈도우로 잘린 데이터를 모으는 리스트
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                if args.shuffle_n and (window_i + 1 != total_window): # 마지막 데이터의 경우는 셔플하지 않는다
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # 슬라이딩 윈도우에서 뒷 부분이 누락될 경우 추가해줌
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))

    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data
