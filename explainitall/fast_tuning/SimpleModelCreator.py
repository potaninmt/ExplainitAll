from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
from .Embedder import GPTEmbedder
import numpy as np
import torch
import torch.nn as layers

from .trainers.DenceKerasTrainer import GPTFastTrainer


# Создание датасета
def get_dataset_dence(txts, embedder, tokenizer, n_layer_index = 'all'):
  list_x = []
  list_y = []
  texts = []

  for txt in txts:
    words = txt.split(' ')
    for i in range(0, len(words), 25):
      texts.append(' '.join(words[i:]))


  for text in texts:
    list_x.append(embedder.get_embs_from_gpt(text, n_layer_index= n_layer_index)[:-1][:1024])
    list_y.append(np.array(tokenizer(text)['input_ids'])[1:])
  
  return np.concatenate(list_x), np.concatenate(list_y)

# Сборка GPT
def GptBuild(trainer, gpt_emb:GPT2Model, tokenizer, y_set, path_to_save = 'gpt_model_new'):
  if torch.cuda.is_available():
        gpt_emb.to('cpu')
        
  w_out_matr = np.dot(trainer.adapter_layer.get_weights()[0], trainer.keras_out_weight)
  w_out_matr = np.transpose(w_out_matr) # Матрица весов

  # Создание gpt
  config_gpt = gpt_emb.config
  new_model = GPT2LMHeadModel(config_gpt)

  # Загрузка обученных весов
  new_model.transformer = gpt_emb
  lm = layers.Linear(config_gpt.n_embd, out_features=config_gpt.vocab_size, bias = False) # Создание слоя
  
  tensor_w = torch.tensor(w_out_matr)
  lm.weight = layers.Parameter(tensor_w) # Инициализация весов

  new_model.lm_head = lm # Замена слоя

  new_model.save_pretrained(path_to_save)
  tokenizer.save_pretrained(path_to_save)
  np.save(f'{path_to_save}/set.data', np.array(y_set))
  
  if torch.cuda.is_available():
        gpt_emb.to('cuda:0')




# Создание сети
class SimpleCreator():
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        main_embedder = GPTEmbedder(tokenizer, model)
        self.gpt_emb = main_embedder.get_new_model(num_layers=-1) 
        self.cut_embedder = GPTEmbedder(self.tokenizer, self.gpt_emb)
        self.trainer = GPTFastTrainer(model)
        self.inp_dim = self.trainer.inp_dim
        self.outp_dim = self.trainer.outp_dim
        

  
    # Обучение
    def train(self, data, lr=0.0003, bs = 64, epochs = 6, val_split = 0.0, save_path = 'new_model', dataset_creator = get_dataset_dence):
        x, y = dataset_creator(data, self.cut_embedder, self.tokenizer)
        net = self.trainer.creat_net()
        self.trainer.train(net, x, y, lr=lr, bs = bs, epochs = epochs, val_split = val_split)
        y_set = list(set(y))
        GptBuild(self.trainer, self.gpt_emb, self.tokenizer, y_set, save_path)
        return net