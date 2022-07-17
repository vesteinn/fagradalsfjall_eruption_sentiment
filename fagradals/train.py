import torch
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from mask_utils import check_keyword, mask_keywords

tokenizer = AutoTokenizer.from_pretrained("vesteinn/XLMR-ENIS-finetuned-sst2")
model1 = XLMRobertaForSequenceClassification.from_pretrained("vesteinn/XLMR-ENIS-finetuned-sst2", num_labels=2)
model = XLMRobertaForSequenceClassification.from_pretrained("vesteinn/XLMR-ENIS-finetuned-sst2", num_labels=3,  ignore_mismatched_sizes=True)


# Convert model from only having two classes, average weights for neutral
with torch.no_grad():
    sd=model1.classifier.out_proj.state_dict()
    sd['weight']=torch.stack([sd['weight'][0], sd['weight'][1], (sd['weight'][0] + sd['weight'][1])/2])
    sd['bias']=torch.stack([sd['bias'][0], sd['bias'][1],  (sd['bias'][0] + sd['bias'][1])/2])
    model.classifier.out_proj.load_state_dict(sd)


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=514)


datafiles = {"train": "../data/train.json", "valid": "../data/valid.json", "tolabel": "../data/tolabel.json"}
dataset = load_dataset('json', data_files=datafiles)

# modify if data is already masked
IS_MASKED = False
if not IS_MASKED:
    dataset = dataset.filter(check_keyword)
    dataset = dataset.map(mask_keywords)

dataset = dataset.map(preprocess_function, batched=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    by_cat_p, by_cat_r, by_cat_f1, _ = precision_recall_fscore_support(labels, preds)
    by_cat = [by_cat_p, by_cat_r, by_cat_f1]
    cat_stats = {}
    for i in range(3):
        cat_stats[i] = {"f1": by_cat[2][i], "precision": by_cat[0][i], "recall": by_cat[1][i]}
    acc = accuracy_score(labels, preds)
    total = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'by_category': cat_stats
    }
    return total


training_args = TrainingArguments(
    f"is_geo_twt_mask_temp",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=10,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed = 42
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.model.save_pretrained("is_geo_twt_mask")
