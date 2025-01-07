class VAE_LSTM(nn.Module):
    '''
    Encoder based on LSTM                          
    num_of_features: the number of expected features in the input x
    hidden_size: the number of features in hidden state h
    num_layers: the number of rnn layers
    '''

    def __init__(self, num_of_features_en, num_of_features_de,
                 sequ_len, hidden_size, dropout,
                 latent_len):
        super().__init__()
        self.num_of_features_en = num_of_features_en
        self.num_of_features_de = num_of_features_de
        self.hidden_size = hidden_size
        #self.num_layers = num_layers
        self.dropout = dropout
        self.latent_len = latent_len
        self.sequ_len = sequ_len
        #self.som_grid = {(i,j):nn.Parameter(torch.unsqueeze(torch.randn(num_feature, dtype=torch.float32, requires_grad=True), 0)*uniform_range) for i in range(num_row) for j in range(num_col)}


        self.encoder_lstm = nn.LSTM(input_size=self.num_of_features_en, hidden_size=self.hidden_size,bidirectional=True, dropout=self.dropout)
        self.hidden2mean_layer1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.hidden2mean_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2mean_layer3 = nn.Linear(self.hidden_size, self.latent_len)
        self.hidden2logvar_layer1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.hidden2logvar_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2logvar_layer3 = nn.Linear(self.hidden_size, self.latent_len)
        self.latent2hidden_layer1 = nn.Linear(self.latent_len, self.hidden_size)
        self.latent2hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.latent2hidden_layer3 = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.latent2cell_layer1 = nn.Linear(self.latent_len, self.hidden_size)
        self.latent2cell_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.latent2cell_layer3 = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.decoder_lstmcell = nn.LSTMCell(input_size=self.num_of_features_de, hidden_size=self.hidden_size*2)
        self.hidden2pred = {i: nn.Linear(self.hidden_size*2, self.num_of_features_en) for i in range(self.sequ_len)}

    def lstm_encoded_hc(self, data_x):
        _, (h_T, c_T) = self.encoder_lstm(data_x)
        #print(h_T)
        return {"encoded hidden state": h_T, "encoded cell state": c_T}

    def encode_mu_logvar(self, h_en):
        h = h_en.view(-1, 2*self.hidden_size)
        mu = self.hidden2mean_layer1(h)
        mu = nn.functional.elu(mu)
        mu = self.hidden2mean_layer2(mu)
        mu = nn.functional.elu(mu)
        mu = self.hidden2mean_layer3(mu)
        log_sig_sqr = self.hidden2logvar_layer1(h)
        log_sig_sqr = nn.functional.elu(log_sig_sqr)
        log_sig_sqr = self.hidden2logvar_layer2(log_sig_sqr)
        log_sig_sqr = nn.functional.elu(log_sig_sqr)
        log_sig_sqr = self.hidden2logvar_layer3(log_sig_sqr)
        std = torch.exp(0.5 * log_sig_sqr)
        return {"mean": mu, "std": std, "log_var": log_sig_sqr}

    def reparameterize(self, mu, std, ep):
        #eps = torch.randn(self.latent_len)
        z = mu + ep * std
        return z

    def decoder(self, latent_z):
        ht = self.latent2hidden_layer1(latent_z)
        ht = nn.functional.elu(ht)
        ht = self.latent2hidden_layer2(ht)
        ht = nn.functional.elu(ht)
        ht = self.latent2hidden_layer3(ht)
        h0 = torch.tanh(ht)
        ct = self.latent2cell_layer1(latent_z)
        ct = nn.functional.elu(ct)
        ct = self.latent2cell_layer2(ct)
        ct = nn.functional.elu(ct)
        ct = self.latent2cell_layer3(ct)
        c0 = torch.tanh(ct)
        pred_data = torch.randn(self.sequ_len,self.num_of_features_en)
        for i in range(self.sequ_len):
            pred = torch.sigmoid(self.hidden2pred[i](h0))
            pred_data[i] = pred
            h_0, c_0 = self.decoder_lstmcell(pred, (h0, c0))
        return pred_data
