# Use GPU 0
workers_id = "0"

# Use GPU 1
#workers_id = "1"

# Use GPU 0 and 1 (Multi-GPU)
#workers_id = "0,1"

import torch

def set_gpus_info(workers_id):

    DEVICE = None
    MULTI_GPU = None # flag to use multiple GPUs
    GPU_ID = None # specify your GPU ids

    if workers_id == 'cpu' or not torch.cuda.is_available():
        GPU_ID = []
        print("check", workers_id, torch.cuda.is_available())
    else:
        GPU_ID = [int(i) for i in workers_id.split(',')]
    if len(GPU_ID) == 0:
        DEVICE= torch.device('cpu')
        MULTI_GPU = False
    else:
        DEVICE= torch.device('cuda:%d' % GPU_ID[0])
        if len(GPU_ID) == 1:
            MULTI_GPU = False
        else:
            MULTI_GPU = True
    
    print("Number of GPUs: ", len(GPU_ID))
    print("Device ", DEVICE)

    return DEVICE, MULTI_GPU, GPU_ID 


DEVICE, MULTI_GPU, GPU_ID  = set_gpus_info(workers_id)

num_workers = len(GPU_ID)

from transformers import VisionEncoderDecoderModel

# You can use your own pre-trained HuggingFace models in this function
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-384", "roberta-base", tie_encoder_decoder=True 
        )

from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Si queremos usar un tokenizador (BPE) entrenado por nosotros:
#tokenizer = RobertaTokenizer.from_pretrained('MY_BPE_FOLDER')

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 20
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader

class Image_Captioning_Dataset(Dataset):
    def __init__(self, partition="train", tokenizer=None, feature_extractor=None):
    
        self.l_captions = []
        self.l_images = []
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        self.image_path = None
        self.caption_path = None
        if partition == "train":
            self.image_path = "./data_ved/Images_train/"
            self.caption_path = "./data_ved/train_captions.txt"
        else:
            self.image_path = "./data_ved/Images_test/"
            self.caption_path = "./data_ved/test_captions.txt"

        # Using readlines()
        file1 = open(self.caption_path, 'r')
        Lines = file1.readlines()
        count = 0

        # Strips the newline character
        for line in Lines:
            count += 1
            line = line.strip()
            captions_images = line.split('|')
            self.l_images.append(self.image_path + captions_images[0])
            self.l_captions.append(captions_images[1])

        self._max_label_len = max([8] + [len(label) for label in self.l_captions])
        print("Loaded ", count, " samples!")

    def __len__(self):
        return len(self.l_images)

    def __getitem__(self, idx):
        
        # Shape (h,w,3) and 1...255
        image = Image.open(self.l_images[idx]).convert("RGB")
        image_tensor: torch.Tensor = self.feature_extractor(image, return_tensors="pt").pixel_values[0]

        
        label = self.l_captions[idx]
        label_tensor = self.tokenizer(
            label,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self._max_label_len,
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "label": label_tensor}



test_dataset = Image_Captioning_Dataset(partition="test", tokenizer=tokenizer, feature_extractor=feature_extractor)
train_dataset = Image_Captioning_Dataset(partition="train", tokenizer=tokenizer, feature_extractor=feature_extractor)

batch_size = 2
test_dataloader = DataLoader(test_dataset, batch_size, num_workers=num_workers)
train_dataloader = DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True)

print("ViT input: ", test_dataset.__getitem__(0)["input"].shape)
print("RoBERTa input: ", test_dataset.__getitem__(0)["label"])


import torch
from torch import nn
from tqdm import tqdm

num_epochs = 25

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)

from transformers import get_scheduler
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

if MULTI_GPU:
    print("Multi gpu ", MULTI_GPU)
    # multi-GPU setting
    model = nn.DataParallel(model, device_ids = GPU_ID)
    model = model.to(DEVICE)
else:
    print("single gpu")
    # single-GPU setting
    model = model.to(DEVICE)

model.train()

for epoch in range(num_epochs):

    epoch_loss = 0 
    print("\nEpoch ", epoch)
    optimizer.zero_grad(set_to_none=True)

    with tqdm(iter(train_dataloader), desc="Training set", unit="batch") as tepoch:
        for batch in tepoch:

            inputs: torch.Tensor = batch["input"].to(DEVICE)
            labels: torch.Tensor = batch["label"].to(DEVICE)

            outputs = model(pixel_values=inputs, labels=labels)

            loss = outputs.loss.to(DEVICE)
            loss = loss.sum().to(DEVICE) # for MULTI-GPU
            loss.backward()

            # Update Optimizer
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            tepoch.set_postfix(loss=loss.data.item())

            epoch_loss += loss.data.item()
                
            del loss, outputs
        
        optimizer.zero_grad(set_to_none=True)
        
        epoch_loss = epoch_loss / len(train_dataloader)

        print("Loss: ", epoch_loss)




####
def predict(
    device, multi_gpu, tokenizer, feature_extractor, model: VisionEncoderDecoderModel, dataloader: DataLoader
) -> tuple[list[tuple[int, str]], list[float]]:

    l_generated_text = []
    if multi_gpu:
        model = model.module # unpackage model from DataParallel
        model = model.to(device)
    else:
        model = model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            
            inputs: torch.Tensor = batch["input"].to(device)

            generated_ids = model.generate(inputs) # no necesita text de entrada porque lo tiene que generar
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            l_generated_text.append(generated_text)

    return l_generated_text

model.eval()
l_generated_text = predict(DEVICE, MULTI_GPU, tokenizer, feature_extractor, model, test_dataloader)
print(l_generated_text)



########################
# PRE-TRAINED WORKING:
# https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder