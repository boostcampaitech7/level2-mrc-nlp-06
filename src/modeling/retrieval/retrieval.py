import sys, os
import argparse
from datasets import load_from_disk
from transformers import TrainingArguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils_common import configurer
from trainer_retrieval import DenseRetrieval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../../../retrieval_sample.yaml')
    parser.add_argument('--task', type=str, default="train")
    args = parser.parse_args()

    config = configurer(args.config)
    pool = config.model.pool=="True"
    metric = config.model.metric
    wiki_path = os.path.join(config.data_path, 'wiki_unique.csv')

    retrieval = DenseRetrieval(config.model.name, pool=pool, metric=metric)

    if args.task=="train":        
        train_dataset = load_from_disk(os.path.join(config.data_path, 'train_dataset'))
        validation_dataset = train_dataset['validation']
        train_dataset = train_dataset['train']

        train_args = TrainingArguments(output_dir=config.train.output_dir,
                                       evaluation_strategy='epoch',
                                       learning_rate=float(config.train.learning_rate),
                                       per_device_train_batch_size=int(config.train.batch_size),
                                       per_device_eval_batch_size=int(config.train.batch_size),
                                       num_train_epochs=int(config.train.epoch),
                                       weight_decay=float(config.train.weight_decay))

        retrieval.train(train_args, data=train_dataset)
        retrieval.save_embeddings(wiki_path, config.train.output_dir)

    if args.task=="validate":
        validation_dataset = load_from_disk(os.path.join(config.data_path, 'train_dataset'))['validation']
        retrieval.validate(validation_dataset, int(config.topk), wiki_path, config.train.output_dir)

    if args.task=="predict":
        test_dataset = load_from_disk(os.path.join(config.data_path, 'test_dataset'))['validation']
        retrieval.retrieve(test_dataset, int(config.topk), wiki_path, config.train.output_dir)