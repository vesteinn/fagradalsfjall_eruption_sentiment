import datetime
from collections import defaultdict
from keywords import KEYWORD_CATS as KEYWORDS
import sys


def get_week(ts):
    d = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    y, w = d.isocalendar()[0:2]
    return f"{y}-{w}"


def read_data(file_name):
    data = []
    with open(file_name) as fileh:
        for line in fileh.readlines():
            pred, txt, ts = line[:-1].split("\t")[:3]
            try:
                week = get_week(ts)
            except:
                continue
            data.append({
                "week": week,
                "time": ts,
                "text": txt,
                "label": pred})
    return data


def hist(data):
    ghist = defaultdict(int)
    rhist = defaultdict(int)
    nhist = defaultdict(int)

    for twt in data:
        if twt["label"] == "Negative":
            rhist[twt["week"]] += 1
        elif twt["label"] == "Positive":
            ghist[twt["week"]] += 1
        elif twt["label"] == "Neutral":
            nhist[twt["week"]] += 1
        else:
            breakpoint()
        
    return rhist, ghist, nhist


def print_hist(rhist, ghist, nhist):

    for w in ghist:
        wr = rhist[w]
        wg = ghist[w]
        wn = nhist[w]
        print(f"{w}\t{wr}\t{wg}\t{wn}")


keywords = ["gos", "skjálft"]
def kw_hist(data, keywords):
    hist_data = {}
    for kw in keywords:
        hist_data[kw] = defaultdict(int)
        hist_data["both"] = defaultdict(int)
    for twt in data:
        kw_found = set()
        for kw in keywords:
            for lkw in KEYWORDS[kw]:
                if lkw in twt["text"].lower():
                    kw_found.add(kw)
        for kw in kw_found:
            hist_data[kw][twt["week"]] += 1
        if len(kw_found) > 1:
            hist_data["both"][twt["week"]] += 1
                
    return hist_data    
         

def print_kw_hist(histdata):
    fkeys = list(histdata.keys())
    print("week\t" + '\t'.join(fkeys))
    for wk in histdata[fkeys[0]].keys():
        record = [wk]
        for kw in fkeys:
            record.append(str(histdata[kw][wk]))
        print("\t".join(record))


def check_keyword(tweet):
    for kw in KEYWORDS:
        for kwl in KEYWORDS[kw]:
            if kwl in tweet["text"].lower():
                return True
    return False


data = read_data(sys.argv[1])
rhist, ghist, nhist = hist(data)
print_hist(rhist, ghist, nhist)
