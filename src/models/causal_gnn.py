"""
Causal & Equilibrium GNN Architecture with Neural Gravity Module.
Extends TradeGNN with Transformer attention, a deep nonlinear gravity prior,
learned gating, and residual learning for hybrid Gravity-GNN predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class NeuralGravityLayer(nn.Module):
    """
    Deep nonlinear gravity model for trade prediction.
    
    Learns: Trade ~ f(GDP_i, GDP_j, Pop_i, Pop_j, Distance, FTA, ...)
    with explicit economic interaction terms and residual-friendly structure.
    
    Input features are constructed from node features (source & target)
    and edge attributes, with economically-motivated transformations:

    """
    
    def __init__(
        self,
        num_node_features: int = 4,
        num_edge_features: int = 10,
        hidden_sizes: list = None,
        dropout: float = 0.2
    ):
        super(NeuralGravityLayer, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        # Gravity-specific input:
        #   node_src (4) + node_tgt (4) = 8 base features
        #   + 3 interaction terms (gdp_product, gdp_ratio, pop_product)
        #   + selected edge features: -distance_log, shared_lang, contiguous, fta = 4
        # Total = 15
        gravity_input_dim = num_node_features * 2 + 3 + 4  # 15
        
        self.input_norm = nn.LayerNorm(gravity_input_dim)
        
        layers = []
        in_dim = gravity_input_dim
        for h_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Compute gravity-based trade score for each edge.
        
        Args:
            x: Node features [num_nodes, num_node_features]
               Indices: [0]=gdp_log, [1]=pop_log, [2]=exports_log, [3]=imports_log
            edge_index: [2, num_edges]
            edge_attr: Edge features [num_edges, num_edge_features]
               Indices: [0]=sentiment, [1]=|sentiment|, [2]=distance_log,
                        [3]=shared_lang, [4]=contiguous, [5]=fta,
                        [6]=sector, [7]=lag1, [8]=lag2, [9]=lag3
        
        Returns:
            gravity_score: [num_edges] scalar gravity estimate per edge
        """
        row, col = edge_index
        
        x_src = x[row]  # [E, 4]
        x_tgt = x[col]  # [E, 4]
        
        # --- Economic interaction features ---
        gdp_src = x_src[:, 0]  # gdp_log
        gdp_tgt = x_tgt[:, 0]
        pop_src = x_src[:, 1]  # pop_log
        pop_tgt = x_tgt[:, 1]
        
        gdp_product = (gdp_src + gdp_tgt).unsqueeze(-1)   # Correct: log(GDP_i * GDP_j)
        gdp_ratio   = (gdp_src - gdp_tgt).unsqueeze(-1)   # [E, 1] (log-space division)
        pop_product  = (pop_src + pop_tgt).unsqueeze(-1)   # Correct: log(Pop_i * Pop_j)
        
        # --- Edge-level gravity features ---
        neg_distance  = -edge_attr[:, 2:3]    # Negate: closer -> higher
        shared_lang   = edge_attr[:, 3:4]
        contiguous    = edge_attr[:, 4:5]
        fta           = edge_attr[:, 5:6]
        
        # --- Assemble gravity input ---
        gravity_input = torch.cat([
            x_src,          # node features
            x_tgt,          # node features
            gdp_product,    # interaction
            gdp_ratio,      # interaction
            pop_product,    # interaction
            neg_distance,
            shared_lang,
            contiguous,
            fta,
        ], dim=1)  # Total: 15
        
        gravity_input = self.input_norm(gravity_input)
        gravity_score = self.mlp(gravity_input).squeeze(-1)  # [E]
        
        return gravity_score


class CausalTradeGNN(nn.Module):
    """
    Advanced Hybrid Gravity-GNN with:
      - NeuralGravityLayer: deep nonlinear gravity prior
      - TransformerConv: directed causal attention on trade graph
      - Learned gating: blends gravity baseline with GNN corrections
      - Equilibrium constraints for counterfactual simulation
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
        
        # ============================================================
        # 1. NEURAL GRAVITY MODULE (replaces single linear layer)
        # ============================================================
        self.gravity_module = NeuralGravityLayer(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_sizes=[128, 64],
            dropout=dropout
        )
        
        # ============================================================
        # 2. TRANSFORMER CONV LAYERS (Causal Attention — unchanged)
        # ============================================================
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
        
        # ============================================================
        # 3. GNN RESIDUAL HEAD (edge prediction MLP)
        # ============================================================
        mlp_in = hidden_dim * 2 + num_edge_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ============================================================
        # 4. LEARNED GATE (blends gravity prior with GNN residual)
        # ============================================================
        # Gate input: GNN edge embedding + gravity score
        gate_in = hidden_dim * 2 + num_edge_features + 1
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr, return_attention=False):
        """
        Forward pass with Hybrid Gravity-GNN architecture.
        """
        x = x.float()
        edge_attr = edge_attr.float()
        
        # ---- Step A: Neural Gravity Prior ----
        gravity_score = self.gravity_module(x, edge_index, edge_attr)
        
        # ---- Step B: Transformer layers (causal attention) ----
        h = x
        alpha = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if return_attention and i == len(self.convs) - 1:
                h, (att_edge_index, att_weights) = conv(h, edge_index, edge_attr, return_attention_weights=True)
                alpha = (att_edge_index, att_weights)
            else:
                h = conv(h, edge_index, edge_attr)
            
            if h.shape[0] > 1:
                h = bn(h)
                
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.elu(h)
        
        # ---- Step C: GNN Residual prediction ----
        row, col = edge_index
        edge_emb = torch.cat([h[row], h[col], edge_attr], dim=1)
        gnn_residual = self.edge_mlp(edge_emb).squeeze(-1)
        
        # ---- Step D: Learned Gating ----
        gate_input = torch.cat([
            h[row], h[col], edge_attr, gravity_score.unsqueeze(-1)
        ], dim=1)
        gate = self.gate(gate_input).squeeze(-1)
        
        # ---- Step E: Hybrid prediction ----
        pred = gate * gravity_score + (1.0 - gate) * gnn_residual
        
        if return_attention:
            return pred, alpha, gate
            
        return pred

    def calculate_equilibrium_loss(self, pred_trade, edge_index, x):
        """
        Calculates trade equilibrium loss.
        Constrains total trade to be proportional to GDP.
        """
        row, col = edge_index
        trade_usd = torch.exp(pred_trade)
        
        num_nodes = x.size(0)
        node_exports = torch.zeros(num_nodes, device=x.device)
        node_imports = torch.zeros(num_nodes, device=x.device)
        
        node_exports.scatter_add_(0, row, trade_usd)
        node_imports.scatter_add_(0, col, trade_usd)
        
        # Budget constraint: predicted trade shouldn't exceed capacity
        gdp_usd = torch.exp(x[:, 0])
        trade_to_gdp_ratio = (node_exports + node_imports) / (gdp_usd + 1e-6)
        
        # Penalize values exceeding unscaled capacity
        budget_violation = torch.relu(trade_to_gdp_ratio - 1.5).mean()
        
        # Global balance check (though T_exp == T_imp is always true by sum)
        global_balance = torch.abs(trade_usd.sum() - trade_usd.sum()) # identity
        
        return budget_violation
