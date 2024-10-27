import os
import sys
import logging
import datasets

# import datasets.metric
import evaluate
import torch.nn as nn

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split


train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
# # 只加载前100个样本 用于验证代码的正确性
# train = train.head(100)
# test = test.head(100)

class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # print("config", config)
        self.num_labels = 2#config.num_labels加载的bert模型config.json文件中没有num_labels属性 只有0和1两个标签
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
# outputs[0]：模型的最后一层的隐藏状态，形状为(batch_size, sequence_length, hidden_size)。
# outputs[1]：池化后的输出，通常是一个形状为(batch_size, hidden_size)的向量，用于表示整个句子的嵌入表示。
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2) # 80% train and 20% validation

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True) #对文本数据进行分词。truncation=True 表示如果文本长度超过分词器的最大长度，将会被截断。

# 获取输入数据
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
#DataCollatorWithPadding 类通常用于将多个样本整理成一个批次（batch），不同样本的长度可能不同，因此需要对它们进行填充（padding）以确保所有样本具有相同的长度。
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertScratch.from_pretrained('bert-base-uncased')

    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=6,  # batch size per device during training
        per_device_eval_batch_size=12,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="steps",  # save model every N steps
        save_steps=1000,  # N steps
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/bert_scratch.csv", index=False, quoting=3)
    logging.info('result saved!')