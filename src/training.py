


from sklearn import model_selection

import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch import nn

import config
import TweetModel
from Tweetdataset import Tweetdataset
import util

class Training:
    def __init__(self):
        self.num_train_steps=0


    def data_processing(self):
        df_train=pd.read_csv(config.TRAINING_FILE)
        X = df_train[['text', 'selected_text', 'sentiment']]
        y = df_train[['sentiment']].values
        df_train_x1, df_test_x1, df_train_y, df_test_y = model_selection.train_test_split(X, y, test_size=0.33,
                                                                                          random_state=42)
        df_train_x = pd.DataFrame(data=df_train_x1, columns=['text', 'selected_text', 'sentiment'])
        df_test_x = pd.DataFrame(data=df_test_x1, columns=['text', 'selected_text', 'sentiment'])
        self.num_train_steps = int(len(df_train_x) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
        train_dataset = Tweetdataset(
            tweet=df_train_x.text.values,
            sentiment=df_train_x.sentiment.values,
            selected_text=df_train_x.selected_text.values
        )

        valid_dataset = Tweetdataset(
            tweet=df_test_x.text.values,
            sentiment=df_test_x.sentiment.values,
            selected_text=df_test_x.selected_text.values
        )

        return train_dataset,valid_dataset

    def data_loading(self,train_dataset,valid_dataset):

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            num_workers=4
        )
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            num_workers=4
        )
        return train_data_loader,valid_data_loader

    def train_fn(self,train_data_loader, model, optimizer, device, scheduler):
        model.train()
        losses = util.AverageMeter()
        jaccards = util.AverageMeter()

        tk0 = tqdm(train_data_loader, total=len(train_data_loader))

        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask_token_type_ids=mask,
            token_type_ids=token_type_ids,
        )
        loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = util.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)



    def loss_fn(self,outputs_start, outputs_end, targets_start, targets_end):
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(outputs_start, targets_start)
        end_loss = loss_fct(outputs_end, targets_end)
        total_loss = (start_loss + end_loss)
        return total_loss

    def eval_fn(self,valid_data_loader, model, device):
        model.eval()
        losses = util.AverageMeter()
        jaccards = util.AverageMeter()

        with torch.no_grad():
            tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))

            for bi, d in enumerate(tk0):
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                targets_start = d["targets_start"]
                targets_end = d["targets_end"]
                sentiment = d["sentiment"]
                orig_selected = d["orig_selected"]
                orig_tweet = d["orig_tweet"]
                targets_start = d["targets_start"]
                targets_end = d["targets_end"]
                offsets = d["offsets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask_token_type_ids=mask,
                token_type_ids=token_type_ids,
            )
            loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)


            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = util.calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
        print(f"Jaccard = {jaccards.avg}")
        return jaccards.avg


    def model_training(self,train_data_loader,valid_data_loader):
        device = torch.device(config.DEVICE)
        model_config = transformers.RobertaConfig.from_pretrained(config.BERT_PATH)
        #model_config.output_hidden_states = True
        model = TweetModel(config=model_config)
        model.to(device)


        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps
        )

        es = util.EarlyStopping(patience=2, mode="max")
        for epoch in range(config.EPOCHS):
            self.train_fn(train_data_loader, model, optimizer, device, scheduler)
            jaccard = self.eval_fn(valid_data_loader, model, device)
            print(f"Jaccard Score = {jaccard}")
            es(jaccard, model, model_path=f"model_{epoch}.bin")
            if es.early_stop:
                print("Early stopping")
                break

    def run(self):
        train_ds,test_ds=self.data_processing()
        train_dl,test_dl=self.data_loading(train_ds,test_ds)
        self.model_training(train_dl,test_dl)

if __name__ == "__main__":
    print('here')





