from happytransformer import GENSettings, HappyGeneration
import numpy as np
import torch
import torch.nn as layers

# Генерация текста
class TextGenerator():
  def __init__(self, path):
    self.generator = HappyGeneration(model_name=path)
    self.device = self.generator.model.device
    linear = self.generator.model.lm_head
    self.y_set = np.load(f'{path}/set.data.npy')

    lm = layers.Linear(linear.in_features,linear.out_features)
    lm.weight = linear.weight
    self.generator.model.lm_head = lm
    self.set_variety_of_answers(0.0)
    self.def_gen_settings = GENSettings(temperature=0.7, no_repeat_ngram_size=2, num_beams=12, top_k=30)
  
  # Установка вариативности
  def set_variety_of_answers(self, variety = 0, min_prob = 3e-3):
    set_tokens = self.y_set
    bias = np.zeros((self.generator.model.lm_head.out_features))
    coef_mask = np.log2(variety+min_prob)/np.log2(np.e)
    bias += coef_mask

    for token in set_tokens:
      bias[token] = 0

    b_tensor = torch.tensor(bias)
    b_param = layers.Parameter(b_tensor)
    self.generator.model.lm_head.bias = b_param
    self.generator.model.to(self.device)
  
  # Генерация
  def generate(self, start_text, args = None):
    if args == None:
      args = self.def_gen_settings
    return self.generator.generate_text(start_text, args=args).text