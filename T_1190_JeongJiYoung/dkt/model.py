import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)

        self.embedding_test = nn.Embedding(self.args.n_cate_cols['testId'] + 1, self.hidden_dim//3) # num_embeddings->size of "dictionary" of embedding (유니크클래스+1) , embedding_dim
        self.embedding_question = nn.Embedding(self.args.n_cate_cols['assessmentItemID'] + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_cate_cols['KnowledgeTag'] + 1, self.hidden_dim//3)
        self.embedding_grade = nn.Embedding(self.args.n_cate_cols['grade'] + 1, self.hidden_dim//3)
        # self.embedding_prior_answerCode = nn.Embedding(self.args.n_cate_cols['prior_answerCode'] + 1, self.hidden_dim//3)

        # embedding combination projection
        self.cate_proj = nn.Linear((self.hidden_dim//3) * (len(args.n_cate_cols) + 1), self.hidden_dim)

        # numeric features
        # self.embedding_numeric = nn.Linear(self.args.n_numeric, self.hidden_dim, bias=False)
        self.embedding_numeric = nn.Sequential(
            nn.Linear(self.args.n_numeric, self.hidden_dim, bias=False),
            # nn.LayerNorm(self.hidden_dim)
        )
        
        # cate + numeric
        self.comb_proj = nn.Linear(self.hidden_dim * 2 , self.hidden_dim)


        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()


    def forward(self, input):
        # big_tag_delta, big_test_delta, big_assess_delta, \
        test, question, tag, grade, \
        prior_elapsed, mean_elapse, test_time, grade_time, \
        answer_delta, \
        tag_delta, test_delta, assess_delta, \
        tag_cumAnswer, \
        _, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_grade = self.embedding_grade(grade)
        # embed_prior_answerCode = self.embedding_prior_answerCode(prior_answerCode)

        cate_embed = torch.cat([
            embed_interaction,
            embed_test,
            embed_question,
            embed_tag,
            embed_grade,
            # embed_prior_answerCode,
            ], 2)

        cate_embed = self.cate_proj(cate_embed)
        
        b_size = prior_elapsed.size(0) # current
        seq_size = prior_elapsed.size(1)

        prior_elapsed = prior_elapsed.view(b_size, seq_size, 1)
        mean_elapse = mean_elapse.view(b_size, seq_size, 1)
        test_time = test_time.view(b_size, seq_size, 1)
        answer_delta = answer_delta.view(b_size, seq_size, 1)
        tag_delta = tag_delta.view(b_size, seq_size, 1)
        test_delta = test_delta.view(b_size, seq_size, 1)
        assess_delta = assess_delta.view(b_size, seq_size, 1)
        # big_tag_delta = big_tag_delta.view(b_size, seq_size, 1)
        # big_test_delta = big_test_delta.view(b_size, seq_size, 1)
        # big_assess_delta = big_assess_delta.view(b_size, seq_size, 1)
        grade_time = grade_time.view(b_size, seq_size, 1)
        tag_cumAnswer = tag_cumAnswer.view(b_size, seq_size, 1)

        embedding_numeric = self.embedding_numeric(torch.cat([
            prior_elapsed, mean_elapse, test_time, answer_delta, 
            tag_delta, test_delta, assess_delta, 
            # big_tag_delta, big_test_delta, big_assess_delta, 
            grade_time, tag_cumAnswer
            ], 2))

        embed = torch.cat([cate_embed,
                           embedding_numeric,
                           ], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds



class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding 
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3) # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(self.args.n_cate_cols['testId'] + 1, self.hidden_dim//3) # num_embeddings->size of "dictionary" of embedding (유니크클래스+1) , embedding_dim
        self.embedding_question = nn.Embedding(self.args.n_cate_cols['assessmentItemID'] + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_cate_cols['KnowledgeTag'] + 1, self.hidden_dim//3)
        self.embedding_grade = nn.Embedding(self.args.n_cate_cols['grade'] + 1, self.hidden_dim//3)
        # self.embedding_prior_answerCode = nn.Embedding(self.args.n_cate_cols['prior_answerCode'] + 1, self.hidden_dim//3)
        
        # embedding combination projection
        self.cate_proj = nn.Linear((self.hidden_dim//3) * (len(args.n_cate_cols) + 1), self.hidden_dim)

        # numeric features
        self.embedding_numeric = nn.Sequential(
            nn.Linear(self.args.n_numeric, self.hidden_dim, bias=False),
            # nn.LayerNorm(self.hidden_dim)
        )
        
        # cate + numeric
        self.comb_proj = nn.Linear(self.hidden_dim * 2 , self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, grade, \
        prior_elapsed, mean_elapse, test_time, answer_delta, tag_delta, test_delta, assess_delta, \
        _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        ## Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_grade = self.embedding_grade(grade)
        # embed_prior_answerCode = self.embedding_prior_answerCode(prior_answerCode)

        cate_embed = torch.cat([
            embed_interaction,
            embed_test,
            embed_question,
            embed_tag,
            embed_grade,
            # embed_prior_answerCode,
            ], 2)

        cate_embed = self.cate_proj(cate_embed)
        
        b_size = prior_elapsed.size(0) # current
        seq_size = prior_elapsed.size(1)

        prior_elapsed = prior_elapsed.contiguous().view(b_size, seq_size, 1)
        mean_elapse = mean_elapse.contiguous().view(b_size, seq_size, 1)
        test_time = test_time.contiguous().view(b_size, seq_size, 1)
        answer_delta = answer_delta.contiguous().view(b_size, seq_size, 1)
        tag_delta = tag_delta.contiguous().view(b_size, seq_size, 1)
        test_delta = test_delta.contiguous().view(b_size, seq_size, 1)
        assess_delta = assess_delta.contiguous().view(b_size, seq_size, 1)

        embedding_numeric = self.embedding_numeric(torch.cat([
            prior_elapsed, mean_elapse, test_time, answer_delta, tag_delta, test_delta, assess_delta
            ], 2))
        
        embed = torch.cat([cate_embed,
                           embedding_numeric,
                           ], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds



class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True) # input_size, hidden_size, num_layers, batch_first(?)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


