import torch.nn as nn
import torch
import torch.nn.functional as F

######### cell embedding models

class TableCellModel_SG(nn.Module):
    def __init__(self, vdim, encdim, hp):
        super(TableCellModel_SG, self).__init__()
        self.vdim = vdim
        
        self.encoder = nn.Sequential(nn.Dropout(p=0.3), 
                                    nn.Linear(vdim, encdim),
                                    # nn.BatchNorm1d(vdim),
                                    nn.Dropout(p=0.3))

        self.decoder = nn.Sequential(nn.Linear(encdim, vdim),
                                     nn.Linear(vdim, vdim))

    def forward(self, target_vecs):
        """
        target_vecs: (batch_size, 1, vdim)
        """
        h1 = target_vecs
        h = h1.view(h1.shape[0], -1)
        ench = self.encoder(h)
        dech = self.decoder(ench)    
        return dech.squeeze(), ench.squeeze()
      
class TableCellModel_CBOW(nn.Module):
    def __init__(self, vdim, encdim, num_context, hp):
        super(TableCellModel_CBOW, self).__init__()
        self.vdim = vdim
        self.encoder = nn.Sequential(nn.Dropout(p=0.3), 
                                    nn.Linear(vdim*num_context, encdim),
                                    # nn.BatchNorm1d(vdim*num_context),
                                    nn.Dropout(p=0.3))
        
        self.decoder = nn.Sequential(nn.Linear(encdim, vdim),
                                     nn.Linear(vdim, vdim))

    def forward(self, context_vecs):
        """
        context_vecs: (batch_size, num_context, vdim)
        """
        h1 = context_vecs
        #h = torch.sum(h1, dim=1) # sum of context vectors
        h = h1.view(h1.shape[0], -1)  # concatenation of context vectors
        ench = self.encoder(h)
        dech = self.decoder(ench)
        return dech.squeeze(), ench.squeeze()

class CEModel(nn.Module):
    def __init__(self, vdim, encdim, num_context, hp=True):
        super(CEModel, self).__init__()
        self.vdim = vdim
        self.encdim = encdim
        self.num_context = num_context

        self.SG = TableCellModel_SG(vdim, encdim, hp)
        self.CBOW = TableCellModel_CBOW(vdim, encdim, num_context, hp)

    def forward(self, context_vecs, target_vecs):
        """
        context_vecs: (batch_size, num_context, vdim)
        target_vecs: (batch_size, 1, vdim)
        """
        et_r, et_emb = self.SG(target_vecs)
        ec_r, ec_emb = self.CBOW(context_vecs)

        return et_emb, ec_emb, et_r, ec_r
################# feature encoding model
class FeatEnc(nn.Module):
    def __init__(self, fdim, encdim):
        super(FeatEnc, self).__init__()
        self.encdim = encdim
        self.fdim = fdim
        self.encoder = nn.Sequential(nn.Linear(fdim, encdim),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(encdim, encdim))
        self.decoder = nn.Sequential(nn.Linear(encdim, fdim),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(fdim, fdim))
    def forward(self, input_vecs):
        ench = self.encoder(input_vecs)
        dech = self.decoder(ench)
        return dech, ench

################# cell classification models
class ClassificationModel(nn.Module):
    def __init__(self, dim, num_tags):
        super(ClassificationModel, self).__init__()
        self.dim1 = dim
        self.reduce_dim = nn.Linear(dim , 100)
        dim = 100
        self.dim = dim
        self.row_lstm = nn.LSTM(input_size=dim,
                                hidden_size=dim, batch_first=True, bidirectional=True)
        self.col_lstm = nn.LSTM(input_size=dim,
                                hidden_size=dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(4 * dim, num_tags)
        self.num_tags = num_tags

    def init_hidden(self, k):
        device = next(self.row_lstm.parameters()).device
        return (torch.zeros(2, k, self.dim).to(device),
                torch.zeros(2, k, self.dim).to(device))
        
    def forward(self, features):
        n,m, d = features.shape
        features = self.reduce_dim(features.view(-1, d)).view(n,m,self.dim)
        self.row_hidden = self.init_hidden(features.shape[0])
        lstm_out, self.row_hidden = self.row_lstm(
            features, self.row_hidden)
        row_tensor = lstm_out

        self.col_hidden = self.init_hidden(features.shape[1])
        lstm_out, self.col_hidden = self.col_lstm(
            features.permute(1, 0, 2), self.col_hidden)
        col_tensor = lstm_out.permute(1, 0, 2)

        table_tensor = torch.cat([row_tensor, col_tensor], dim=2)
        tag_space = self.hidden2tag(table_tensor)
        log_probs = F.log_softmax(tag_space, dim=2)
        
        return log_probs

