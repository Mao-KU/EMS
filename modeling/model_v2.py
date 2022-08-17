import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
import logging
import copy
import math
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertLaEmbedding(nn.Module):
    def __init__(self, la_num, embedding_size, dropout=0.0):
        """
        Args:
            vocab_size: scale, the number of tokens
            embedding_size: scale, the feature size of the token
            d_model: the size of model. need to re-scale the embedding
            padding_idx: 
        """
        super(BertLaEmbedding, self).__init__()
        self.embedding = nn.Embedding(la_num, embedding_size)
        self.layer_norm = BertLayerNorm(embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        tensor = torch.cuda.LongTensor if inputs.is_cuda else torch.LongTensor
        output = self.embedding(tensor(inputs))
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output
       
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=0):
        """
        Args:
            vocab_size: scale, the number of tokens
            embedding_size: scale, the feature size of the token
            d_model: the size of model. need to re-scale the embedding
            padding_idx: 
        """
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        
    def forward(self, inputs):
        tensor = torch.cuda.LongTensor if inputs.is_cuda else torch.LongTensor
        seq_embedding = self.embedding(tensor(inputs))
        return seq_embedding
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """
        
        Args:
            d_model: a scale. dimension of model. 512 default.
            max_seq_len: a scale. max sequence length.
        """
        super(PositionalEncoding, self).__init__()
        # set padding index = 0
        self.position_embeddings = nn.Embedding(max_seq_len+1, d_model, 0)
        
    def forward(self, input_ids, mask):
        """
        Args:
          input_len: a tensor，shape [BATCH_SIZE, 1] represent the true length of input sequences
        Returns:
          aligned positional encoding.
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(1, 1+seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)*mask.type_as(position_ids)
        return self.position_embeddings(position_ids)
    
    
class SentEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_seq_len, dropout=0.0, padding_idx=0):
        super(SentEmbedding, self).__init__()
        self.word_embedding = WordEmbedding(vocab_size, embedding_size, padding_idx)
        self.position_embedding = PositionalEncoding(embedding_size, max_seq_len)
        self.layer_norm = BertLayerNorm(embedding_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, mask):
        word_emb = self.word_embedding(input_ids)
        position_emb = self.position_embedding(input_ids, mask)
        output = word_emb + position_emb
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output
    

class BertSelfAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(BertSelfAttention, self).__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (model_dim, num_heads))
        self.num_attention_heads = num_heads
        self.attention_head_size = int(model_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(model_dim, self.all_head_size)
        self.key = nn.Linear(model_dim, self.all_head_size)
        self.value = nn.Linear(model_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # attention_probs shape: [batch_size, num_heads, seq_len, seq_len]
        # value_layer shape: [batch_size, num_heads, seq_len, dim_per_head]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    
class BertSelfOutput(nn.Module):
    def __init__(self, model_dim=256, dropout=0.0):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(model_dim, model_dim)
        self.LayerNorm = BertLayerNorm(model_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, atten_dropout=0.0, hidden_dropout=0.0):
        super(BertAttention, self).__init__()
        self.self_attention = BertSelfAttention(model_dim, num_heads, atten_dropout)
        self.output = BertSelfOutput(model_dim, hidden_dropout)

    def forward(self, input_tensor, attention_mask):
        # multi-head attention
        self_output = self.self_attention(input_tensor, attention_mask)
        # add & norm 
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, model_dim=512, intermediate_size=2048, ffn_act='gelu'):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(model_dim, intermediate_size)
        self.intermediate_act_fn = ACT2FN[ffn_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self, model_dim=512, intermediate_size=2048, hidden_dropout=0.0):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, model_dim)
        self.LayerNorm = BertLayerNorm(model_dim, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states, input_tensor):
        # feed-forward part
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # add & norm part
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    
class BertLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, intermediate_size=2048, ffn_act='gelu', \
                 atten_dropout=0.0, hidden_dropout=0.0):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(model_dim,num_heads, atten_dropout, hidden_dropout)
        self.intermediate = BertIntermediate(model_dim, intermediate_size, ffn_act)
        self.output = BertOutput(model_dim, intermediate_size, hidden_dropout)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    
class BertEncoder(nn.Module):
    def __init__(self, num_layers=1, model_dim=512, num_heads=8, intermediate_size=2048, ffn_act='gelu', \
                 atten_dropout=0.0, hidden_dropout=0.0):
        super(BertEncoder, self).__init__()
        layer = BertLayer(model_dim, num_heads, intermediate_size, ffn_act, \
                 atten_dropout, hidden_dropout)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i,layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, model_dim, pool_method='MEAN'):
        super(BertPooler, self).__init__()
        self.pooling = pool_method
        
    def count_masked_sum(self, output, mask):
        masked = output.mul(mask.unsqueeze(-1).repeat(1,1, output.shape[-1]).type_as(output))
        return masked

    def forward(self, output, mask):
        # output shape: [batch_size, seq_len, feature_dim]
        masked = self.count_masked_sum(output, mask)
        # masked shape: [batch_size, seq_len, feature_dim]
        if self.pooling=='MAX':
            output,_ = torch.max(masked, dim=1)
        elif self.pooling=='SUM':
            output = torch.sum(masked, dim=1)
        elif self.pooling=='MEAN':
            output = torch.sum(masked, dim=1)
            lens = torch.sum(mask, dim=1).unsqueeze(-1).clone().detach()
            output = torch.true_divide(output, lens)
        return output
    
class BertModel(nn.Module):
    def __init__(self, vocab_size, config):
        super(BertModel, self).__init__()
        self.embedding = SentEmbedding(vocab_size, \
                                       config.token_dim, \
                                       config.max_seq_len,\
                                       config.emb_dropout)
        self.encoder = BertEncoder(config.num_layers, \
                                   config.token_dim, \
                                   config.num_heads,\
                                   config.ffn_dim,\
                                   config.ffn_act, \
                                   config.attention_dropout, \
                                   config.hidden_dropout)
        self.pooler = BertPooler(config.token_dim, config.pooling_method)
        
    def forward(self, input_ids, attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -100000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        embedding_output = self.embedding(input_ids, attention_mask)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output, attention_mask)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
    
    
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, model_dim=512, hidden_act='tanh', change_dim_to=-1, dropout=0.0):
        super(BertPredictionHeadTransform, self).__init__()
        if change_dim_to == -1:
            self.dense = nn.Linear(model_dim, model_dim)
            self.LayerNorm = BertLayerNorm(model_dim, eps=1e-12)
        else:
            self.dense = nn.Linear(model_dim, change_dim_to)
            self.LayerNorm = BertLayerNorm(change_dim_to, eps=1e-12)
        self.transform_act_fn = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    
class BertLMPredictionHead(nn.Module):
    def __init__(self, la_dim, token_dim, vocab_size, token_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if la_dim != 0 and token_dim != 0: # has_la_emb, not share_emb
            self.decoder = nn.Linear(la_dim + token_dim, vocab_size, bias=False)
        elif la_dim == 0 and token_dim != 0: # not has_la_emb, not share_emb
            self.decoder = nn.Linear(token_dim, vocab_size, bias=False)
        else: # share_emb
            self.decoder = nn.Linear(token_embedding_weights.size(1), \
                    token_embedding_weights.size(0), bias=False)
            self.decoder.weight = token_embedding_weights
            vocab_size = token_embedding_weights.size(0)
        # self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states) # + self.bias
        return hidden_states
 
    def inference(self, hidden_states):
        return hidden_states
    
class BertForMaskedLM(nn.Module):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    ```
    """
    def __init__(self, vocab_size, config):
        super(BertForMaskedLM, self).__init__()
        self.vocab_size = vocab_size
        self.bert = BertModel(vocab_size, config)
        self.share_emb = config.share_emb
        self.has_la_emb = config.has_la_emb
        if self.has_la_emb and self.share_emb:
            cat_dim = config.token_dim + config.la_dim
            self.la_tag_linear = BertLaEmbedding(
                    config.la_num, 
                    config.la_dim, 
                    config.emb_dropout,
                    )
            self.transform = BertPredictionHeadTransform(cat_dim, 
                    config.hiddent_act, 
                    change_dim_to=config.token_dim,
                    dropout=config.xtr_dropout,
                    )
            self.target_embedding = self.bert.embedding.word_embedding.embedding
            self.decoder = BertLMPredictionHead(0, 0, 0, self.target_embedding.weight)
        elif self.has_la_emb and not self.share_emb:
            cat_dim = config.token_dim + config.la_dim
            self.la_tag_linear = BertLaEmbedding(
                    config.la_num, 
                    config.la_dim, 
                    config.emb_dropout,
                    )
            self.transform = BertPredictionHeadTransform(
                    cat_dim, 
                    config.hiddent_act,
                    dropout=config.xtr_dropout,
                    )
            self.decoder = BertLMPredictionHead(config.la_dim, config.token_dim, config.vocab_size)
        elif not self.has_la_emb and self.share_emb:
            self.transform = BertPredictionHeadTransform(
                    config.token_dim, 
                    config.hiddent_act,
                    dropout=config.xtr_dropout,
                    )
            self.target_embedding = self.bert.embedding.word_embedding.embedding
            self.decoder = BertLMPredictionHead(0, 0, 0, self.target_embedding.weight)
        else:
            self.transform = BertPredictionHeadTransform(
                    config.token_dim, 
                    config.hiddent_act,
                    dropout=config.xtr_dropout,
                    )
            self.decoder = BertLMPredictionHead(0, config.token_dim, config.vocab_size)

    def forward(self, input_ids, input_len, tgt_la):
        batch_size = input_len.shape[0]
        max_len = input_ids.shape[1]
        attention_mask = (torch.arange(0, max_len,device=input_len.device)\
                          .type_as(input_len)\
                          .expand(batch_size, max_len).lt(input_len.unsqueeze(-1)))
        _, sent_emb = self.bert(input_ids, attention_mask, output_all_encoded_layers=False)
        
        if self.has_la_emb:
            la_emb = self.la_tag_linear(tgt_la)
            cat_la_output = torch.cat((la_emb, sent_emb), dim=1)
        else:
            cat_la_output = sent_emb
        
        transformed_output = self.transform(cat_la_output)
        prediction_scores = self.decoder(transformed_output)
        
        return prediction_scores, sent_emb

    def inference(self, input_ids, input_len):
        batch_size = input_len.shape[0]
        max_len = input_ids.shape[1]
        attention_mask = (torch.arange(0, max_len,device=input_len.device)\
                          .type_as(input_len)\
                          .expand(batch_size, max_len).lt(input_len.unsqueeze(-1)))
        _, sent_emb = self.bert(input_ids, attention_mask, output_all_encoded_layers=False)

        return sent_emb

def calc_accuracy(products,label):
    max_vals, max_indices = torch.max(products,1)
    acc = (max_indices == label).sum().type_as(products)/max_indices.size()[0]
    return acc

def calc_recall(products,label):
    TP = 0
    labels = torch.nonzero(label)
    for i in range(labels.shape[0]):
        if products[labels[i][0]][labels[i][1]] > 0:
            TP += 1
    recall = torch.true_divide(TP, labels.shape[0])
    return recall


class Log1PlusExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        return x.where(torch.isinf(exp), exp.log1p())
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 + (-x).exp())
    
class BilingualModel(nn.Module):
    def __init__(self, vocab_size, config):
        super(BilingualModel, self).__init__()
        self.vocab_size = vocab_size
        self.model = BertForMaskedLM(vocab_size, config)
        self.classification_loss = nn.KLDivLoss(reduction="batchmean")
        self.has_sentence_loss = config.has_sentence_loss
        self.has_sentence_similarity_loss = config.has_sentence_similarity_loss
        self.loss_w1 = config.sentence_alignment_loss_weight
        self.loss_w2 = config.sentence_similarity_loss_weight
        # self.neg_samples = config.neg_samples
        # self.generative_task = config.generative_task
        if self.has_sentence_loss:
            self.log_1_plus_exp = Log1PlusExp.apply
            self.crossentropy = nn.CrossEntropyLoss(reduction='mean')
            self.contrastive_method = config.contrastive_method
            self.temperature = config.contrastive_T
            self.alignproj = nn.Sequential(
                    nn.Linear(config.sent_dim, config.sent_dim),
                    nn.ReLU(),
                    nn.Dropout(config.sent_dropout),
                    nn.Linear(config.sent_dim, 128),
                    )
        if self.has_sentence_similarity_loss:
            self.softmax = nn.Softmax(dim=1)

    # @amp.autocast()
    def forward(self, input_ids1, input_len1, input_ids2, input_len2, 
            masked_lm_label, masked_la, tgt_la1, tgt_la2):
        self.device = input_ids1.device
        label1 = torch.zeros(input_ids1.shape[0], self.vocab_size, device=self.device)
        label2 = torch.zeros(input_ids1.shape[0], self.vocab_size, device=self.device)
        # """
        for i in range(label1.shape[0]):
            label1[i].index_fill_(0, input_ids2[i][: input_len2[i]], torch.true_divide(1., input_len2[i]))
            label2[i].index_fill_(0, input_ids1[i][: input_len1[i]], torch.true_divide(1., input_len1[i]))
        """
        for i, j in enumerate(masked_la):
            if j == 1: # mask language 1
                label1[i][masked_lm_label[i]] = 0.5
                label1[i].index_fill_(0, input_ids2[i][: input_len2[i]], 
                        torch.true_divide(0.5, input_len2[i]))
                label2[i].index_fill_(0, input_ids1[i][: input_len1[i]], 
                        torch.true_divide(1., input_len1[i]))
            elif j == 2: # mask language 2
                label2[i][masked_lm_label[i]] = 0.5
                label1[i].index_fill_(0, input_ids2[i][: input_len2[i]], 
                        torch.true_divide(1., input_len2[i])) 
                label2[i].index_fill_(0, input_ids1[i][: input_len1[i]], 
                        torch.true_divide(0.5, input_len1[i]))
            else:
                raise ValueError
        """
        pred_scores1, sent_emb1 = self.model(input_ids1, input_len1, tgt_la1)
        pred_scores2, sent_emb2 = self.model(input_ids2, input_len2, tgt_la2)
        pred_scores1 = nn.LogSoftmax(dim=1)(pred_scores1)
        pred_scores2 = nn.LogSoftmax(dim=1)(pred_scores2)
        loss = xtr_loss = self.classification_loss(pred_scores1, label1) \
                + self.classification_loss(pred_scores2, label2)
        sentalign_loss = torch.tensor(0., device=self.device, dtype=loss.dtype)
        acc = torch.tensor(0., device=self.device, dtype=loss.dtype)
        if self.has_sentence_loss:
            sent_emb1 = self.alignproj(sent_emb1)
            sent_emb2 = self.alignproj(sent_emb2)
            if self.contrastive_method == "dot":
                sim = torch.mm(sent_emb1, sent_emb2.t())
            elif self.contrastive_method == "cos":
                batch_size = sent_emb1.shape[0]
                emb_size = sent_emb1.shape[1]
                sim = nn.CosineSimilarity(dim=-1)(sent_emb1.expand((batch_size, batch_size, emb_size)), 
                        torch.transpose(sent_emb2.expand((batch_size, batch_size, emb_size)), 0, 1))
            else:
                raise ValueError
            sim = sim / self.temperature
            sim_t = torch.transpose(sim, 0, 1)
            label = torch.arange(0, sent_emb1.shape[0], dtype=torch.long, device=self.device)	
            sentalign_loss = (self.crossentropy(sim, label) 
                    + self.crossentropy(sim_t, label)) / 2
            # loss = 0. * loss + self.loss_w1 * sentalign_loss
            loss = loss + self.loss_w1 * sentalign_loss
        if self.has_sentence_similarity_loss: 
            sent_emb1 = self.alignproj(sent_emb1)
            sent_emb2 = self.alignproj(sent_emb2)
            mono_products_1_softmax = self.softmax(torch.mm(sent_emb1, sent_emb1.t()))
            mono_products_2_softmax = self.softmax(torch.mm(sent_emb2, sent_emb2.t()))
            sentsim_loss = torch.sum(torch.log(torch.cos((mono_products_1_softmax 
                - mono_products_2_softmax) * math.pi)) * (-1)) / ((sent_emb1.size()[0] - 1) ** 2)
            loss = loss + self.loss_w2 * sentsim_loss
        return loss, xtr_loss, sentalign_loss, acc
   
    def inference(self, input_ids, input_len):
        return self.model.inference(input_ids, input_len)
