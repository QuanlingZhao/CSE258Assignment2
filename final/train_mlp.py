import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.functional import relu
from collections import defaultdict, OrderedDict
from sklearn import metrics
from ray.air import session

# hyperparameters
NUM_TARGETS= 3

USER_PATHWAY  = [256, 128, 64]
ITEM_PATHWAY = [256, 128, 64]
COMBINED_PATHWAY = [256, 128, 64, 16]

EMBED_DIM = 10
NUM_ITEM_EMBED = 1378
NUM_USER_EMBED = 47958
NUM_CUPSIZE_EMBED =  12
NUM_CATEGORY_EMBED = 7

NUM_USER_NUMERIC = 5
NUM_ITEM_NUMERIC = 2

DROPOUT = 0.3

EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 128

class ModCloth(torch.utils.data.Dataset):
    def __init__(self,datapath):
        self.data = pd.read_csv(datapath)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        r = self.data.iloc[idx,:]

        return {
            "user_id" : np.array(r['user_id'], dtype=np.int64),
            "cup_size" : np.array(r['cup_size'], dtype=np.int64),
            "user_numeric" : np.array([r['waist'], r['hips'], r['bra_size'], r['height'], r['shoe_size']], dtype=np.float32),
            "item_id" : np.array(r['item_id'], dtype = np.int64),
            "category" :np.array(r['category'], dtype = np.int64),
            "item_numeric" : np.array([r['size'], r['quality']], dtype=np.float32),
            "fit" : np.array(r['fit'], dtype=np.int64)
        }

datasets = OrderedDict()
splits = ['train', 'valid']
datasets['train'] =  ModCloth("data/modcloth_final_data_processed_train.csv")
datasets['valid'] =  ModCloth("data/modcloth_final_data_processed_valid.csv")
datasets['test'] = ModCloth("data/modcloth_final_data_processed_test.csv")


# macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account
# weighted - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

def compute_metrics(target, pred_probs, averaging = "macro"):

    pred_labels = pred_probs.argmax(-1)
    precision = metrics.precision_score(target, pred_labels, average=averaging)
    recall = metrics.recall_score(target, pred_labels, average=averaging)
    f1_score = metrics.f1_score(target, pred_labels, average=averaging)
    accuracy = metrics.accuracy_score(target, pred_labels)
    auc = metrics.roc_auc_score(target, pred_probs, average=averaging, multi_class="ovr")

    return precision, recall, f1_score, accuracy, auc

class Base(nn.Module):
    def __init__(self, user_pathway, item_pathway, combined_pathway, embed_dim, num_item_embed, num_user_embed, num_cupsize_embed, num_category_embed, dropout):
        super().__init__()
       
        self.user_pathway = user_pathway
        self.item_pathway = item_pathway
        self.combined_pathway = combined_pathway
        self.embedding_dim = embed_dim

        self.user_embedding = nn.Embedding(num_user_embed, embed_dim, max_norm=1.0 )
        self.cup_size_embedding = nn.Embedding(num_cupsize_embed, embed_dim, max_norm=1.0 )
        self.item_embedding = nn.Embedding(num_item_embed, embed_dim, max_norm=1.0 )
        self.category_embedding = nn.Embedding(num_category_embed, embed_dim, max_norm=1.0 )


    def forward(self, batch_input):
        # Customer Pathway
        user_emb = self.user_embedding(batch_input["user_id"])
        cup_size_emb = self.cup_size_embedding(batch_input["cup_size"])
        user_representation = torch.cat( [user_emb, cup_size_emb, batch_input["user_numeric"]], dim=-1 )
        user_representation = self.user_transform_blocks(user_representation)

        # Article Pathway
        item_emb = self.item_embedding(batch_input["item_id"])
        category_emb = self.category_embedding(batch_input["category"])
        item_representation = torch.cat( [item_emb, category_emb, batch_input["item_numeric"]], dim=-1 )
        item_representation = self.item_transform_blocks(item_representation)

        # Combine the pathways
        combined_representation = torch.cat( [user_representation, item_representation], dim=-1 )
        combined_representation = self.combined_blocks(combined_representation)

        # Output layer of logits
        logits = self.hidden2output(combined_representation)
        pred_probs = F.softmax(logits, dim=-1)

        return logits, pred_probs
    
class MLP(Base):
    def __init__(self,user_pathway, item_pathway, combined_pathway, embed_dim, num_item_embed, num_user_embed, num_cupsize_embed, num_category_embed, dropout):
        super().__init__(user_pathway, item_pathway, combined_pathway, embed_dim, num_item_embed, num_user_embed, num_cupsize_embed, num_category_embed, dropout)

        # Customer pathway transformation  ==  user_embedding_dim + cup_size_embedding_dim + num_user_numeric_features
        user_features_input_size = 2 * self.embedding_dim + NUM_USER_NUMERIC
        self.user_pathway.insert(0, user_features_input_size)
        self.user_transform_blocks = []
        for i in range(1, len(self.user_pathway)):
            self.user_transform_blocks.append( LinearBlock( self.user_pathway[i - 1], self.user_pathway[i] ) )
        self.user_transform_blocks = nn.Sequential(*self.user_transform_blocks)

        # Article pathway transformation == item_embedding_dim + category_embedding_dim + num_item_numeric_features
        item_features_input_size = 2 * self.embedding_dim + NUM_ITEM_NUMERIC
        self.item_pathway.insert(0, item_features_input_size)
        self.item_transform_blocks = []
        for i in range(1, len(self.item_pathway)):
            self.item_transform_blocks.append( LinearBlock( self.item_pathway[i - 1], self.item_pathway[i])  )
        self.item_transform_blocks = nn.Sequential(*self.item_transform_blocks)

        # Combined top layer pathway
        # u = output dim of user_transform_blocks, # t = output dim of item_transform_blocks
        # Pathway combination through [u, t] # Hence, input dimension will be 4*dim(u)
        combined_layer_input_size = 2 * self.user_pathway[-1]
        self.combined_pathway.insert(0, combined_layer_input_size)
        self.combined_blocks = []
        for i in range(1, len(self.combined_pathway)):
            self.combined_blocks.append( LinearBlock( self.combined_pathway[i - 1], self.combined_pathway[i]) )
        self.combined_blocks = nn.Sequential(*self.combined_blocks)

        # Linear transformation from last hidden layer to output
        self.hidden2output = nn.Linear(self.combined_pathway[-1], NUM_TARGETS)


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        """ Skip Connection for feed-forward  - ResNet Block """
        super().__init__()
        self.W1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """  z = ReLU(   W2( ReLU( W1(x))) + Projection(x))    """
        return relu(self.W1(x))

def train(config):
    model = MLP(config['user_pathway'], config['item_pathway'], config['combined_pathway'], config['embed_dim'], config['num_item_embed'], config['num_user_embed'],config['num_cupsize_embed'],config['num_category_embed'],config['dropout'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], weight_decay= config['wd'])
    
    step = 0

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    for epoch in range(config['epochs']):

        for d in datasets:
            for split in splits:
                data_loader = DataLoader( dataset=datasets[split], batch_size=config['batch_size'], shuffle = (split == "train") )
                loss_tracker = defaultdict(tensor)

                # Enable/Disable Dropout
                if split == "train":
                    model.train()
                else:
                    model.eval()
                    target_tracker = []
                    pred_tracker = []

                for iteration, batch in enumerate(data_loader):

                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    # Forward pass
                    logits, pred_probs = model(batch)

                    # loss calculation
                    loss = loss_criterion(logits, batch["fit"])   # batch['fit'] are the true labels

                    # backward + optimization
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step += 1

                    # bookkeepeing
                    loss_tracker["Total Loss"] = torch.cat((loss_tracker["Total Loss"], loss.view(1)))

         

                    if split == "valid":
                        target_tracker.append(batch["fit"].cpu().numpy())
                        pred_tracker.append(pred_probs.cpu().data.numpy())

    target_tracker = []
    pred_tracker = []

  

    data_loader = DataLoader(dataset = datasets['test'], batch_size=config['batch_size'], shuffle=False)

 
    model.eval()
    with torch.no_grad():

        for iteration, batch in enumerate(data_loader):

            for k, v in batch.items():
                batch[k] = v.to(device)

            # Forward pass
            _, pred_probs = model(batch)

            target_tracker.append(batch["fit"].cpu().numpy())
            pred_tracker.append(pred_probs.cpu().data.numpy())

    target_tracker = np.stack(target_tracker[:-1]).reshape(-1)
    pred_tracker = np.stack(pred_tracker[:-1], axis=0).reshape(-1, config['num_targets'])
    precision, recall, f1_score, accuracy, auc = compute_metrics(target_tracker, pred_tracker, averaging = "weighted")
    session.report({'precision':precision,'recall':recall,'f1_score':f1_score,'accuracy':accuracy,'auc':auc})
    #tune.track(precision=precision)

def main(num_samples):
    config={'embed_dim':tune.choice([5,10,15,20]),
           'lr':tune.choice([0.0005,0.001,0.0015,0.002]),
           'dropout':tune.choice([0.2,0.4,0.6]),
           'wd':tune.choice([0.00005,0.0001,0.0004,0.0008]),
           'num_targets':3,
           'user_pathway':[256, 128, 64],
           'item_pathway':[256, 128, 64],
           'combined_pathway':[256, 128, 64,16],
           'num_item_embed':1378,
           'num_user_embed':47958,
           'num_cupsize_embed':12,
           'num_category_embed':7,
           'num_user_numeric':5,
           'num_item_numeric':2,
           'batch_size':128,
           'epochs':tune.choice([2,5,10])}
    print(config['epochs'])
    analysis=tune.run(train,config=config,num_samples=num_samples)
    df=analysis.dataframe()
    df.to_csv('results.csv')

main(num_samples=50)

