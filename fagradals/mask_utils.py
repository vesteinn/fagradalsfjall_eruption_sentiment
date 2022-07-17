from keywords import KEYWORDS


def mask_keywords(example):
    new_text = []
    for w in example["text"].split():
        masked = False
        for kw in KEYWORDS:
            if kw in w.lower():
                new_text.append("<mask>")
                masked = True
                break
        if not masked:
            new_text.append(w)
    return {
        "label": example["label"],
        "text": " ".join(new_text),
        "original_text": example["text"]
    }


def check_keyword(tweet):
    for kw in KEYWORDS:
        if kw in tweet["text"].lower():
            return True
    return False
