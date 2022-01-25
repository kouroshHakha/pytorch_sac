import torch
import random
import numpy as np
import matplotlib.pyplot as plt

import transformers
from decision_transformer.models.trajectory_gpt2 import GPT2Model

hidden_dim = 128
seq_len = 32
seed = 10

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

gpt_config = transformers.GPT2Config(
    vocab_size=1,
    n_embd=hidden_dim,
    n_layer=1,
    n_head=1,
    # n_inner=h_dim,
    activation_function='tanh',
    n_positions=2048,
    resid_pdrop=0,
    attn_pdrop=0,
    embd_pdrop=0,
    summary_first_dropout=0,
)

model = GPT2Model(gpt_config)

input_tokens = torch.randn(1, seq_len, hidden_dim)
input_tokens.requires_grad = True
# mask = torch.ones(1, 10).long()
outputs = model(inputs_embeds=input_tokens, output_attentions=True) #, attention_mask=mask)
output_tokens = outputs['last_hidden_state']
attentions = (outputs['attentions'][0][0,0]).detach().numpy()

# output_token_scores = output_tokens.squeeze(0).sum(-1)
#
# grads = [torch.autograd.grad(output_token_scores[i], input_tokens, retain_graph=True)[0].squeeze(0).sum(-1) for i in range(len(output_token_scores))]
# grads = torch.stack(grads, 0) # output i to input j
#
# attn = (grads.abs() > 0).int()
# plt.imshow(attn)
plt.imshow(attentions)
plt.show()

# import torch.nn as nn
#
# nn.MultiheadAttention()