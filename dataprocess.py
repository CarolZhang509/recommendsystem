import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# test
train_path = './data/train.txt'
test_path = './data/test.txt'
idx_path = './data/node_idx.pkl'
train_user_pkl = './data/train_user.pkl'
train_item_pkl = './data/train_item.pkl'
bx_pkl = './data/bx.pkl'
bi_pkl = './data/bi.pkl'
user_num = 19835
item_num = 455691
ratings_num = 5001507

def get_idx(train_path):
    all_nodes = set()
    with open(train_path, 'r') as f:
        while (line := f.readline()) != '':
            _, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id, _ = map(int, line.strip().split())
                all_nodes.add(item_id)
    node_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    return node_idx

def get_train_data(train_path, node_idx):
    data_user, data_item = defaultdict(list), defaultdict(list)
    with open(train_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id, score = line.strip().split()
                item_id, score = int(item_id), float(score)
                score = score / 10  # 把0-100的得分映射到0-10
                data_user[user_id].append([node_idx[item_id], score])
                data_item[node_idx[item_id]].append([user_id, score])
    return data_user, data_item

def get_test_data(test_path):
    data = defaultdict(list)
    with open(test_path, 'r') as f:
        while (line := f.readline()) != '':
            user_id, num = map(int, line.strip().split('|'))
            for _ in range(num):
                line = f.readline()
                item_id = int(line.strip())
                data[user_id].append(item_id)
    return data

def split_data(data_user, ratio=0.85, shuffle=True):
    train_data, valid_data = defaultdict(list), defaultdict(list)
    for user_id, items in data_user.items():
        if shuffle:
            np.random.shuffle(items)
        train_data[user_id] = items[:int(len(items) * ratio)]
        valid_data[user_id] = items[int(len(items) * ratio):]
    return train_data, valid_data

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def store_data(pkl_path, data):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def get_bias(train_data_user, train_data_item):
    miu = 0.0
    bx = np.zeros(user_num, dtype=np.float64)
    bi = np.zeros(item_num, dtype=np.float64)
    for user_id in train_data_user:
        sum = 0.0
        for item_id, score in train_data_user[user_id]:
            miu += score
            sum += score
        bx[user_id] = sum / len(train_data_user[user_id])
    miu /= ratings_num

    for item_id in train_data_item:
        sum = 0.0
        for user_id, score in train_data_item[item_id]:
            sum += score
        bi[item_id] = sum / len(train_data_item[item_id])

    bx -= miu
    bi -= miu
    return miu, bx, bi

if __name__ == '__main__':
    print('Start to process data...')
    node_idx = get_idx(train_path)
    with open(idx_path, 'wb') as f:
        pickle.dump(node_idx, f)

    user_data, item_data = get_train_data(train_path, node_idx)
    store_data(train_user_pkl, user_data)
    store_data(train_item_pkl, item_data)

    test_data = get_test_data(test_path)
    store_data(test_path.replace('.txt', '.pkl'), test_data)
    print('Data processing done!')

    print('Loading data...')
    train_user_data = load_pkl(train_user_pkl)
    train_item_data = load_pkl(train_item_pkl)
    print('Data loaded.')

    miu, bx, bi = get_bias(train_user_data, train_item_data)

    print('Saving data...')
    store_data(bx_pkl, bx)
    store_data(bi_pkl, bi)
    print('Data saved.')

    print('评分均值：', miu)
