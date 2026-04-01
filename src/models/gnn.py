"""
GNN Architecture - Updated for your feature dimensions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TradeGNN(nn.Module):
    """Graph Attention Network for Trade Prediction"""
    
    def __init__(
        self,
        num_node_features: int = 4,  # gdp_log, pop_log, exports, imports
        num_edge_features: int = 10,  # Updated for your features
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4
    ):
        super(TradeGNN, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(
                num_node_features, 
                hidden_dim, 
                heads=heads, 
                concat=True, 
                dropout=dropout
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * heads, 
                    hidden_dim, 
                    heads=heads, 
                    concat=True, 
                    dropout=dropout
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Last layer (single head)
        self.convs.append(
            GATConv(
                hidden_dim * heads, 
                hidden_dim, 
                heads=1, 
                concat=False, 
                dropout=dropout
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge prediction MLP
        mlp_in = hidden_dim * 2 + num_edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass"""
        # Ensure float32
        x = x.float()
        edge_attr = edge_attr.float()
        
        # Node learning through GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            
            # Batch norm (handle single sample case)
            if x.shape[0] > 1:
                x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.elu(x)
        
        # Edge prediction
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col], edge_attr], dim=1)
        
        # MLP prediction
        pred = self.edge_mlp(edge_emb)
        
        return pred.squeeze(-1)