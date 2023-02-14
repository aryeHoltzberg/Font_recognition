from train import Train

if __name__ == '__main__':
    T = Train('datasets/datasets_06')
    T.eval_model('results/final/models/top_5')
