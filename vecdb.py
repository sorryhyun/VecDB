from datasets import load_from_disk, Dataset
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import re, time, duckdb, json
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from accelerate import Accelerator

class VecDict(SentenceEvaluator):
    def __init__(self, sent_emb_model : SentenceTransformer):
        accelerator = Accelerator()
        self.model = sent_emb_model
        self.device = accelerator.device

    def reset_model(self, new_model : SentenceTransformer):
        self.model = new_model

    def set_labels_for_entity_text(self, target_label):
        self.entity_target_label = target_label

    def set_vec_dict(self, entity_text_to_embed):
        self.entity_text_to_embed = entity_text_to_embed
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.entity_embeds = self.model.encode(self.entity_text_to_embed, 
                                               convert_to_tensor=True)
        self.model = self.model.train()
    
    def set_vec_dict_inference(self, entity_text_to_embed):
        self.entity_text_to_embed = entity_text_to_embed
        self.entity_embeds = self.model.encode(self.entity_text_to_embed, 
                                               convert_to_tensor=True)

    def set_query_answers(self, query_answer_pairs): # query and answers (all text)
        self.query_answer_pairs = []
        self.query_text_to_embed = [x[0] for x in query_answer_pairs]
        for answers_in_text in [x[1] for x in query_answer_pairs]:
            temp_answers = []
            for answer_text in answers_in_text:
                temp_answers.append(self.entity_text_to_embed.index(answer_text))
            self.query_answer_pairs.append(temp_answers)

    def __call__(self, model=None, output_path=None, epoch=-1, steps=-1):
        self.reset_model(model)
        return self.return_topk(only_topk=True)

    def return_topk(self, only_topk=False):
        self.entity_embeds = self.model.encode(self.entity_text_to_embed, convert_to_tensor=True)
        query_embeds = self.model.encode(self.query_text_to_embed, convert_to_tensor=True)
        dataloader = DataLoader(query_embeds, batch_size=128)

        indices = {'top3':[], 'top5':[], 'top10':[]}
        with torch.no_grad():
            for query_embed in dataloader:
                similarities = query_embed @ self.entity_embeds.T

                _, top3_inds = torch.topk(similarities, k=3, dim=1, largest=True, sorted=True)
                _, top5_inds = torch.topk(similarities, k=5, dim=1, largest=True, sorted=True)
                _, top10_inds = torch.topk(similarities, k=10, dim=1, largest=True, sorted=True)
                indices['top3'].extend(top3_inds.cpu().tolist())
                indices['top5'].extend(top5_inds.cpu().tolist())
                indices['top10'].extend(top10_inds.cpu().tolist())

        answer_and_preds = {'questions':[],'answers':[], 
                            'preds10':[], 'pred_ids10':[], 
                            'preds3':[], 'pred_ids3':[]}
        scores = {'top3':0, 'top5':0, 'top10':0}
        for k_num in ['top3', 'top5', 'top10']:
            for idx in range(len(self.query_answer_pairs)):
                if set(self.query_answer_pairs[idx]) & set(indices[k_num][idx]):
                    scores[k_num]+=1/len(self.query_answer_pairs)

        for idx in range(len(self.query_answer_pairs)):
            answer_and_preds['questions'].append(self.query_text_to_embed[idx])
            temp_answers = []
            for answer in self.query_answer_pairs[idx]:
                temp_answers.append(self.entity_text_to_embed[answer])
            answer_and_preds['answers'].append(temp_answers)

            temp_preds10 = []
            temp_pred_ids10 = []
            for pred in indices['top10'][idx]:
                temp_preds10.append(self.entity_text_to_embed[pred])
                temp_pred_ids10.append(pred)
            answer_and_preds['preds10'].append(temp_preds10)
            answer_and_preds['pred_ids10'].append(temp_pred_ids10)

            temp_preds3 = []
            temp_pred_ids3 = []
            for pred in indices['top3'][idx]:
                temp_preds3.append(self.entity_text_to_embed[pred])
                temp_pred_ids3.append(pred)
            answer_and_preds['preds3'].append(temp_preds3)
            answer_and_preds['pred_ids3'].append(temp_pred_ids3)
            
        if only_topk:
            return scores
        else:
            return scores, answer_and_preds
