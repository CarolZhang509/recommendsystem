import pickle
import numpy as np
from collections import defaultdict

train_path = './data/train.txt'
test_path = './data/test.txt'
attribute_path = './data/itemAttribute.txt'
idx_path = './data/node_idx.pkl'

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
                # 把0-100的得分映射到0-10
                score = score / 10
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

if __name__ == '__main__':
    print('Start to process data...')
    node_idx = get_idx(train_path)
    with open(idx_path, 'wb') as f:
        pickle.dump(node_idx, f)
    
    user_data, item_data = get_train_data(train_path, node_idx)
    store_data(train_path.replace('.txt', '_user.pkl'), user_data)
    store_data(train_path.replace('.txt', '_item.pkl'), item_data)

    test_data = get_test_data(test_path)
    store_data(test_path.replace('.txt', '.pkl'), test_data)
    print('Done!')
