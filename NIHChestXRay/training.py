import torch
import os
import shutil
import numpy as np
import pandas as pd
import contextlib
import io
from pathlib import Path
from scipy.special import softmax
import json
import matplotlib.pyplot as plt

from torchvision import transforms
import transformers
import datasets

# ChestXRay-14 dataset

dataset_rootdir = Path("./").absolute()

# Path to the extracted "images" directory
images_dir = dataset_rootdir / "images"

# Path to Data_Entry_2017_v2020.csv
label_file = dataset_rootdir / 'Data_Entry_2017_v2020.csv'

data = pd.read_csv(label_file)

# Converts the format of each label in the dataframe from "LabelA|LabelB|LabelC"
# into ["LabelA", "LabelB", "LabelC"], concatenates the
# lists together and removes duplicate labels
unique_labels = np.unique(
    data['Finding Labels'].str.split("|").aggregate(np.concatenate)
).tolist()

print(f"Dataset contains the following labels:\n{unique_labels}")


#transform the labels into N-hot encoded arrays

label_index = {v: i for i, v in enumerate(unique_labels)}


def string_to_N_hot(string: str):
    true_index = [label_index[cl] for cl in string.split("|")]
    label = np.zeros((len(unique_labels),), dtype=float)
    label[true_index] = 1
    return label

data["labels"] = data["Finding Labels"].apply(string_to_N_hot)

data[["Image Index", "labels"]].rename(columns={"Image Index": "file_name"}).to_json(images_dir / 'metadata.jsonl', orient='records', lines=True)


train_val_split = 0.05
dataset = datasets.load_dataset(
    "imagefolder",
    data_dir=images_dir,
)

split = dataset["train"].train_test_split(train_val_split)
dataset["train"] = split["train"]
dataset["validation"] = split["test"]

model_name_or_path = "google/vit-base-patch16-224-in21k"

feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
    model_name_or_path
)


class XRayTransform:
    """
    Transforms for pre-processing XRay data across a batch.
    """
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.Resize(feature_extractor.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ])

    def __call__(self, example_batch):
        example_batch["pixel_values"] = [self.transforms(pil_img) for pil_img in example_batch["image"]]
        return example_batch


# Set the training transforms
dataset["train"].set_transform(XRayTransform())
# Set the validation transforms
dataset["validation"].set_transform(XRayTransform())

def batch_sampler(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


print(data.head(10))

fig = plt.figure(figsize=(20, 15))

unique_labels = np.array(unique_labels)


for i, data_dict in enumerate(dataset['validation']):
    if i == 12:
        break
    image = data_dict["pixel_values"]
    label = data_dict["labels"]
    ax = plt.subplot(3, 4, i + 1)
    ax.set_title(", ".join(unique_labels[np.argwhere(label).flatten()]))
    plt.imshow(image[0])  # Plot only the first channel as they are all identical

fig.tight_layout()


model = transformers.AutoModelForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(unique_labels)
    )


metric_auc = datasets.load_metric("roc_auc", "multilabel")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)

    pred_scores = softmax(p.predictions.astype('float32'), axis=1)
    auc = metric_auc.compute(prediction_scores=pred_scores, references=p.label_ids, multi_class='ovo')['roc_auc']
    return {"roc_auc": auc}


training_args = transformers.TrainingArguments(
    output_dir="./vit-base-patch16-224-in21k",
    remove_unused_columns=False,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    num_train_epochs=100,
    weight_decay=0.01,
    learning_rate=1e-4,
    warmup_steps=500,
    logging_dir="./vit-base-patch16-224-in21k/logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=batch_sampler,
)

trainer.train()

trainer.evaluate()


