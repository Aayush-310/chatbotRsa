from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import pandas as pd


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Load your training dataset from a CSV file
train_data = pd.read_csv("train.csv")  # Replace with your actual CSV file path

# Extract text and labels from the training dataset
train_texts = train_data["text"].tolist()
train_labels = train_data["category"].tolist()

# Define your training dataset
train_dataset = {
    "text": train_texts,
    "label": train_labels,
}

# Load your evaluation dataset from a CSV file
eval_data = pd.read_csv("eval.csv")  # Replace with your actual CSV file path

# Extract text and labels from the evaluation dataset
eval_texts = eval_data["text"].tolist()
eval_labels = eval_data["category"].tolist()

# Define your evaluation dataset
eval_dataset = {
    "text": eval_texts,
    "label": eval_labels,
}
# Fine-tune BERT on your dataset
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    output_dir="./bert_finetuned_model",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model()
