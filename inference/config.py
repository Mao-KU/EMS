class DefaultConfigs:
    config_path = 'config.py'
    pooling_method = 'MEAN'
    ffn_act = 'swish'
    hiddent_act = 'swish'
    vocab_size = 60000
    la_num = 62
    max_seq_len = 120
    la_dim = 128
    lower = True
    has_la_emb = True
    share_emb = False
    contrastive_method = "cos"
    sentence_alignment_loss_weight = 2.
    sentence_similarity_loss_weight = 0.
    eval_batch_size = 32
    
    def set_eval(self, resume=False, is_train=False):
        self.emb_dropout = 0.0
        self.attention_dropout = 0.0
        self.hidden_dropout = 0.0
        self.xtr_dropout = 0.0
        self.sent_dropout = 0.0
        self.bpe_path = '../preprocessing/62languages_bpe_60000.model'
        self.output_dir = "../ckpt/"

    def set_config(self, model_name, resume=False, is_train=True):
        self.emb_dropout = 0.0
        self.attention_dropout = 0.0
        self.hidden_dropout = 0.0
        self.xtr_dropout = 0.0
        self.sent_dropout = 0.0
        self.bpe_path = '../preprocessing/62languages_bpe_60000.model'
        self.output_dir = "../ckpt/"
        self.model_name = model_name
        
        if model_name == 'EMS':
            self.num_layers = 6
            self.sent_dim = 1024
            self.token_dim = 1024
            self.num_heads = 16
            self.ffn_dim = 4096
            self.train_batch_size = 152
            self.vali_batch_size = 32
            self.has_sentence_loss = True
            self.has_sentence_similarity_loss = False
            self.contrastive_T = 0.1
        else:
            print('No such model %s! Please check.' % model_name)
            raise

config = DefaultConfigs()

