import os
import sys
import json
from pprint import pprint
from collections import Counter

# hard voting 방식의 ensemble


def get_predictions(path):

    predictions_per_id = {}
    print(" ***** Predictions for Ensemble ***** ")
    # predictions_{ver}.json들 가져오기
    for names in os.listdir(save_dir):
        data_path = os.path.join(save_dir, names)
        print(names)
        with open(data_path, "r") as f:
            pred_data = json.load(f)
        for id in pred_data:
            if id in predictions_per_id:
                predictions_per_id[id].append(pred_data[id])
            else:
                predictions_per_id[id] = [pred_data[id]]

    return predictions_per_id


def get_preds_json(preds, result_dir):
    ensemble = {}
    for id, answers in preds.items():
        counts = Counter(answers)
        ensemble[id] = counts.most_common(1)[0][0]
        print(counts)
    # with open(os.path.join(result_dir,"ensemble.json"),"w") as f:
    #     json.dump(ensemble, f, ensure_ascii=False, indent=4)

    print(" ***** ensemble task complete ***** ")


if __name__ == "__main__":
    save_dir = os.path.join(os.getcwd(), "predictions", "ensemble", "pred")
    result_dir = os.path.join(os.getcwd(), "predictions", "ensemble")
    preds = get_predictions(save_dir)
    get_preds_json(preds, result_dir)
