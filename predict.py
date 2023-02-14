from train import Train as Model
from results_writer import ResultsWriter
from tqdm import tqdm
import argparse
from create_dataset import SPLITTER2 ,PATH_SPLITTER


def main(args):
    model = Model(args.data_path,predict_mode= True)
    print('start predicting')
    predicts,labels = model.predict(args.model_path)
    results_writer = ResultsWriter('final_results')
    print('write results to file')
    for i in (tqdm(range(len(predicts)))):
        img_name, word =  labels[i].split(SPLITTER2)
        word = word.replace(PATH_SPLITTER,'/')
        results_writer.add_word(img_name,word,predicts[i])
    results_writer.close()
    print('finish')
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model prediction')
    parser.add_argument('--model_path',action='store',dest='model_path',default = 'results/final/models/top_5',type=str)
    parser.add_argument('--data_path',action='store',dest='data_path',default = 'submit_file/dataset',type=str)
    main(parser.parse_args())