import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from dataprocess import split_data, load_pkl, store_data

test_pkl = './data/test.pkl'
bx_pkl = './data/bx.pkl'
bi_pkl = './data/bi.pkl'
idx_pkl = './data/node_idx.pkl'


class SVD:
    def __init__(self, model_path='./model', data_path='./data/train_user.pkl', lr=5e-3, 
                 lamda1=1e-2, lamda2=1e-2, lamda3=1e-2, lamda4=1e-2, factor=50):
        self.model_path = model_path
        self.lr = lr
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.lamda4 = lamda4
        self.factor = factor

        # Load data and initialize parameters
        self.bx = load_pkl(bx_pkl)
        self.bi = load_pkl(bi_pkl)
        self.idx = load_pkl(idx_pkl)
        self.train_user_data = load_pkl(data_path)
        self.train_data, self.valid_data = split_data(self.train_user_data)
        self.test_data = load_pkl(test_pkl)
        self.globalmean = self.calculate_globalmean()

        # Initialize P and Q matrices
        self.P = np.random.normal(0, 0.1, (self.factor, len(self.bx)))
        self.Q = np.random.normal(0, 0.1, (self.factor, len(self.bi)))

    def calculate_globalmean(self):
        total_score, total_count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                total_score += score
                total_count += 1
        return total_score / total_count

    def predict(self, user_id, item_id):
        return (self.globalmean + self.bx[user_id] + self.bi[item_id] + 
                np.dot(self.P[:, user_id], self.Q[:, item_id]))

    def compute_loss(self, is_valid=False):
        loss, count = 0.0, 0
        data = self.valid_data if is_valid else self.train_data
        for user_id, items in data.items():
            for item_id, score in items:
                loss += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        if not is_valid:
            loss += (self.lamda1 * np.sum(self.P ** 2) + self.lamda2 * np.sum(self.Q ** 2) +
                     self.lamda3 * np.sum(self.bx ** 2) + self.lamda4 * np.sum(self.bi ** 2))
        return loss / count

    def compute_rmse(self):
        rmse, count = 0.0, 0
        for user_id, items in self.train_user_data.items():
            for item_id, score in items:
                rmse += (score - self.predict(user_id, item_id)) ** 2
                count += 1
        return np.sqrt(rmse / count)

    def train(self, epochs=10, save=False, load=False):
        if load:
            self.load_weights()
        print('Start training...')
        for epoch in range(epochs):
            for user_id, items in tqdm(self.train_data.items(), desc=f'Epoch {epoch + 1}'):
                for item_id, score in items:
                    error = score - self.predict(user_id, item_id)
                    self.bx[user_id] += self.lr * (error - self.lamda3 * self.bx[user_id])
                    self.bi[item_id] += self.lr * (error - self.lamda4 * self.bi[item_id])
                    self.P[:, user_id] += self.lr * (error * self.Q[:, item_id] - self.lamda1 * self.P[:, user_id])
                    self.Q[:, item_id] += self.lr * (error * self.P[:, user_id] - self.lamda2 * self.Q[:, item_id])
            train_loss = self.compute_loss()
            valid_loss = self.compute_loss(is_valid=True)
            print(f'Epoch {epoch + 1} train loss: {train_loss:.6f} valid loss: {valid_loss:.6f}')
        print('Training finished.')
        if save:
            self.save_weights()

    def test(self, write_path='./result/result.txt', load=True):
        if load:
            self.load_weights()
        print('Start testing...')
        predictions = defaultdict(list)
        for user_id, items in self.test_data.items():
            for item_id in items:
                if item_id not in self.idx:
                    predicted_score = self.globalmean * 10
                else:
                    new_id = self.idx[item_id]
                    predicted_score = self.predict(user_id, new_id) * 10
                    predicted_score = np.clip(predicted_score, 0, 100)
                predictions[user_id].append((item_id, predicted_score))
        print('Testing finished.')
        self.write_predictions(predictions, write_path)
        return predictions

    def write_predictions(self, predictions, write_path):
        print('Start writing...')
        with open(write_path, 'w') as f:
            for user_id, items in predictions.items():
                f.write(f'{user_id}|6\n')
                for item_id, score in items:
                    f.write(f'{item_id} {score}\n')
        print('Writing finished.')

    def load_weights(self):
        print('Loading weights...')
        self.bx = load_pkl(os.path.join(self.model_path, 'bx.pkl'))
        self.bi = load_pkl(os.path.join(self.model_path, 'bi.pkl'))
        self.P = load_pkl(os.path.join(self.model_path, 'P.pkl'))
        self.Q = load_pkl(os.path.join(self.model_path, 'Q.pkl'))
        print('Weights loaded.')

    def save_weights(self):
        print('Saving weights...')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        store_data(os.path.join(self.model_path, 'bx.pkl'), self.bx)
        store_data(os.path.join(self.model_path, 'bi.pkl'), self.bi)
        store_data(os.path.join(self.model_path, 'P.pkl'), self.P)
        store_data(os.path.join(self.model_path, 'Q.pkl'), self.Q)
        print('Weights saved.')


if __name__ == '__main__':
    svd = SVD()
    svd.train(epochs=10, save=True, load=False)
    svd.test(write_path='./result/svd.txt')
    # rmse = svd.rmse()
    # print(f'RMSE: {rmse:.6f}')
