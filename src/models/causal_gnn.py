"""
Causal & Equilibrium GNN Architecture
Extends TradeGNN with Transformer attention and Gravity-Hybrid capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class CausalTradeGNN(nn.Module):
    """
    Advanced GNN with Causal Attention (Transformer) 
    and Equilibrium Constraints for Counterfactual Simulation.
    """
    
    def __init__(
        self,
        num_node_features: int = 4,
        num_edge_features: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4
    ):
        super(CausalTradeGNN, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 1. GRAVITY PRIOR: Learned linear mapping for the classic Gravity Model
        # Trade ~ GDP_i * GDP_j / Distance
        self.gravity_linear = nn.Linear(num_node_features * 2 + num_edge_features, 1)
        
        # 2. TRANSFORMER CONV LAYERS (Causal Attention)
        # These replace GATConv for more stable directed attention
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(
            TransformerConv(
                num_node_features, 
                hidden_dim, 
                heads=heads, 
                dropout=dropout,
                edge_dim=num_edge_features,
                concat=True
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Middle layers
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(
                TransformerConv(
                    hidden_dim * heads, 
                    hidden_dim, 
                    heads=heads, 
                    dropout=dropout,
                    edge_dim=num_edge_features,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Last layer
        self.convs.append(
            TransformerConv(
                hidden_dim * heads, 
                hidden_dim, 
                heads=1, 
                dropout=dropout,
                edge_dim=num_edge_features,
                concat=False
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 3. EDGE PREDICTION MLP (Intervention Head)
        mlp_in = hidden_dim * 2 + num_edge_features + 1 # +1 for gravity prior
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass with Causal Attention logic
        """
        # Ensure float tensors
        x = x.float()
        edge_attr = edge_attr.float()
        
        # Step A: Calculate Gravity Prior
        row, col = edge_index
        gravity_input = torch.cat([x[row], x[col], edge_attr], dim=1)
        gravity_prior = self.gravity_linear(gravity_input)
        
        # Step B: Transformer layers (learns directed graph dependencies)
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index, edge_attr)
            
            if h.shape[0] > 1:
                h = bn(h)
                
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.elu(h)
        
        # Step C: Edge Prediction combining GNN hidden state and Gravity Prior
        edge_emb = torch.cat([h[row], h[col], edge_attr, gravity_prior], dim=1)
        pred = self.edge_mlp(edge_emb)
        
        return pred.squeeze(-1)

    def calculate_equilibrium_loss(self, pred_trade, edge_index, num_nodes):
        """
        EQUILIBRIUM CONSTRAINT: sum(exports) should equal sum(imports) globally.
        Ensures the model adheres to basic economic conservation laws.
        """
        row, col = edge_index
        
        # Predicted exports for each source node
        exports = torch.zeros(num_nodes, device=pred_trade.device)
        exports.scatter_add_(0, row, torch.exp(pred_trade)) # using exp since trade is in log scale
        
        # Predicted imports for each target node
        imports = torch.zeros(num_nodes, device=pred_trade.device)
        imports.scatter_add_(0, col, torch.exp(pred_trade))
        
        # Global imbalance loss (Sum of Exports - Sum of Imports should be 0)
        total_exports = torch.sum(exports)
        total_imports = torch.sum(imports)
        
        global_imbalance = torch.abs(total_exports - total_imports) / (total_exports + 1e-6)
        
        # Local imbalance loss (optional: penalty for countries with wild trade deficits)
        # diff = torch.abs(exports - imports)
        
        return global_imbalance
