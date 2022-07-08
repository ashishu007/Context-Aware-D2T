import random
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
random.seed(42)


class BertDataLoader:
    def __init__(self, downsample=False) -> None:
        self.downsample = downsample

    def downsample_data(self, data_list):
        pos = [x for x in data_list if x['label'] == 1]
        neg = [x for x in data_list if x['label'] == 0]
        neg1 = random.sample(neg, len(pos))
        new_data_list = pos + neg1
        return new_data_list

    def prep_data(self, all_data):
        data = {}
        for part in ['train', 'validation', 'test']:
            temp = [{"label": labs, "text": ftrs} for ftrs, labs in tqdm(zip(all_data[part]['ftrs'], all_data[part]['labs']))]
            if self.downsample and part == 'train':
                temp = self.downsample_data(temp)
            data[part] = temp

        train = datasets.Dataset.from_pandas(pd.DataFrame(data['train']))
        valid = datasets.Dataset.from_pandas(pd.DataFrame(data['validation']))
        test = datasets.Dataset.from_pandas(pd.DataFrame(data['test']))

        dataset = datasets.DatasetDict({"train": train, "validation": valid, "test": test})
        train_x = [x['text'] for x in dataset["train"]]
        train_y = np.array([y['label'] for y in dataset["train"]])
        val_x = [x['text'] for x in dataset["validation"]]
        val_y = np.array([y['label'] for y in dataset["validation"]])
        test_x = [x['text'] for x in dataset["test"]]
        test_y = np.array([y['label'] for y in dataset["test"]])

        return dataset, train_x, train_y, val_x, val_y, test_x, test_y


class BertThemeClassifier:
    def __init__(self, num_labels=2) -> None:
        self.model_name = 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {'f1': f1}

    def predict(self, trained_model, text_data, batch_size=32):
        # batches = [[item['text'] for item in text_data[i:i+batch_size]] for i in range(0, len(text_data), batch_size)]
        batches = [text_data[i:i+batch_size] for i in range(0, len(text_data), batch_size)]
        preds = []
        for batch in tqdm(batches):
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = trained_model(**inputs)
            probs = outputs[0].softmax(1)
            preds.extend(list(np.argmax(probs.detach().numpy(), axis=1)))
        return preds

    def train(self, dataset, epochs=3, batch_size=32, lr=1e-5, weight_decay=0.01, warmup_steps=0):
        """
        train_data and validation_data are datasets.Dataset objects
        """

        tokenized_data = dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir='./bert-results',          # output directory
            num_train_epochs=10,              # total number of training epochs
            learning_rate=2e-5,                # learning rate
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=20,   # batch size for evaluation
            warmup_steps=100,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            # logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            metric_for_best_model='f1',     # metric to use for saving best model
            # logging_steps=100,               # log & save weights each logging_steps
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            evaluation_strategy="steps",     # evaluate each `logging_steps`
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.evaluate()

        return trainer.model, self.tokenizer

