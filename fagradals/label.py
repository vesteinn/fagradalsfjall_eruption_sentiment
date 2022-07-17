from transformers import pipeline
import tqdm
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from datasets import load_dataset
from mask_utils import check_keyword, mask_keywords


tokenizer = AutoTokenizer.from_pretrained("vesteinn/XLMR-ENIS-finetuned-sst2")
model = XLMRobertaForSequenceClassification.from_pretrained("./is_geo_twt_mask", num_labels=3)


LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive",
    "LABEL_2": "Neutral"
}


classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
datafiles = {"train": "../data/train.json", "valid": "../data/valid.json", "tolabel": "../data/tolabel.json"}
dataset = load_dataset('json', data_files=datafiles)


# modify if data is already masked
IS_MASKED = False
if not IS_MASKED:
    dataset = dataset.filter(check_keyword)
    dataset = dataset.map(mask_keywords)

for subset in dataset:
    for line in dataset[subset]:
        text = line["text"]
        ts = line["timestamp"]
        classif = classifier(text)
        clab = LABEL_MAP[classif[0]['label']]
        print(f"{clab}\t{text}\t{ts}")