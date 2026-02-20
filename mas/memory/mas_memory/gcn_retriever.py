"""
GCN Retriever Module for G-Memory++
Uses Graph Convolutional Networks to learn node embeddings for improved retrieval.
Supports heterogeneous graphs with query, insight, and skill nodes.
"""

import os
import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. GCN features will be disabled.")

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data, HeteroData
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: PyTorch Geometric not available. GCN features will use fallback.")


# ================================ Node Types ================================

class NodeType:
    QUERY = "query"
    INSIGHT = "insight"
    SKILL = "skill"


# ================================ GCN Models ================================

if HAS_TORCH:
    
    class SimpleGCN(nn.Module):
        """Simple GCN for homogeneous graphs."""
        
        def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim) if HAS_TORCH_GEOMETRIC else None
            self.conv2 = GCNConv(hidden_dim, output_dim) if HAS_TORCH_GEOMETRIC else None
            
            # Fallback if no PyG
            if not HAS_TORCH_GEOMETRIC:
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
            if HAS_TORCH_GEOMETRIC and edge_index is not None:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
            else:
                # Fallback: just use MLPs
                x = self.fc1(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.fc2(x)
            
            return x
    
    
    class GraphSAGE(nn.Module):
        """GraphSAGE for better inductive learning."""
        
        def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
            super().__init__()
            if HAS_TORCH_GEOMETRIC:
                self.conv1 = SAGEConv(input_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, output_dim)
            else:
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
            if HAS_TORCH_GEOMETRIC and edge_index is not None:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
            else:
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
            
            return x
    
    
    class MemoryGraphEncoder(nn.Module):
        """
        Encoder for the memory graph that handles multiple node types.
        Outputs embeddings that can be used for similarity-based retrieval.
        """
        
        def __init__(
            self, 
            text_embed_dim: int = 384,  # Typical for sentence transformers
            goal_feature_dim: int = 25,  # From StructuredGoal.to_features()
            hidden_dim: int = 128,
            output_dim: int = 64,
            model_type: str = "sage"  # "gcn", "sage", "gat"
        ):
            super().__init__()
            
            # Input projection for different feature types
            self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
            self.goal_proj = nn.Linear(goal_feature_dim, hidden_dim // 2)
            
            # Combined input dimension
            combined_dim = hidden_dim + hidden_dim // 2
            
            # GNN layers
            if model_type == "gcn" and HAS_TORCH_GEOMETRIC:
                self.conv1 = GCNConv(combined_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)
            elif model_type == "sage" and HAS_TORCH_GEOMETRIC:
                self.conv1 = SAGEConv(combined_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, output_dim)
            elif model_type == "gat" and HAS_TORCH_GEOMETRIC:
                self.conv1 = GATConv(combined_dim, hidden_dim, heads=4, concat=False)
                self.conv2 = GATConv(hidden_dim, output_dim, heads=1)
            else:
                # Fallback MLP
                self.fc1 = nn.Linear(combined_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
            
            self.model_type = model_type
            self.output_dim = output_dim
        
        def forward(
            self, 
            text_features: torch.Tensor,
            goal_features: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                text_features: (N, text_embed_dim) text embeddings
                goal_features: (N, goal_feature_dim) goal features (optional)
                edge_index: (2, E) edge indices
            
            Returns:
                (N, output_dim) node embeddings
            """
            # Project text features
            x = self.text_proj(text_features)
            
            # Add goal features if available
            if goal_features is not None:
                goal_x = self.goal_proj(goal_features)
                x = torch.cat([x, goal_x], dim=-1)
            else:
                # Pad with zeros
                padding = torch.zeros(x.size(0), self.goal_proj.out_features, device=x.device)
                x = torch.cat([x, padding], dim=-1)
            
            # GNN or MLP
            if HAS_TORCH_GEOMETRIC and edge_index is not None:
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.conv2(x, edge_index)
            else:
                x = self.fc1(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.fc2(x)
            
            # Normalize for cosine similarity
            x = F.normalize(x, p=2, dim=-1)
            
            return x


# ================================ Training Data ================================

@dataclass
class RetrievalExample:
    """A single training example for the retrieval model."""
    
    query_text_embed: np.ndarray
    query_goal_features: Optional[np.ndarray]
    positive_idx: int  # Index of positive (relevant) node
    negative_idxs: List[int]  # Indices of negative nodes
    label: float = 1.0  # Success signal


@dataclass
class GraphData:
    """Stores graph structure and node features."""
    
    node_ids: List[str] = field(default_factory=list)  # Unique node identifiers
    node_types: List[str] = field(default_factory=list)  # NodeType for each node
    text_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    goal_features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Edge list: (source_idx, target_idx, edge_type)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    
    # Metadata for each node
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_node(
        self, 
        node_id: str, 
        node_type: str, 
        text_embedding: np.ndarray,
        goal_feature: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """Add a node to the graph."""
        if node_id in self.node_ids:
            return self.node_ids.index(node_id)
        
        idx = len(self.node_ids)
        self.node_ids.append(node_id)
        self.node_types.append(node_type)
        
        # Convert to numpy array if needed
        if isinstance(text_embedding, list):
            text_embedding = np.array(text_embedding)
        text_embedding = np.asarray(text_embedding).flatten()
        
        if len(self.text_embeddings) == 0:
            self.text_embeddings = text_embedding.reshape(1, -1)
        else:
            self.text_embeddings = np.vstack([self.text_embeddings, text_embedding])
        
        if goal_feature is not None:
            # Convert to numpy array if needed
            if isinstance(goal_feature, list):
                goal_feature = np.array(goal_feature)
            goal_feature = np.asarray(goal_feature).flatten()
            
            if len(self.goal_features) == 0:
                self.goal_features = goal_feature.reshape(1, -1)
            else:
                self.goal_features = np.vstack([self.goal_features, goal_feature])
        
        self.metadata.append(metadata or {})
        
        return idx
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "related"):
        """Add an edge between nodes."""
        if source_id not in self.node_ids or target_id not in self.node_ids:
            return
        
        source_idx = self.node_ids.index(source_id)
        target_idx = self.node_ids.index(target_id)
        self.edges.append((source_idx, target_idx, edge_type))
    
    def get_edge_index(self) -> Optional[np.ndarray]:
        """Get edge index in PyG format: (2, E)."""
        if not self.edges:
            return None
        
        sources = [e[0] for e in self.edges]
        targets = [e[1] for e in self.edges]
        
        return np.array([sources, targets], dtype=np.int64)
    
    def get_node_idx(self, node_id: str) -> int:
        """Get index of a node by ID."""
        if node_id in self.node_ids:
            return self.node_ids.index(node_id)
        return -1
    
    def save(self, path: str):
        """Save graph data to disk."""
        data = {
            "node_ids": self.node_ids,
            "node_types": self.node_types,
            "text_embeddings": self.text_embeddings.tolist() if len(self.text_embeddings) > 0 else [],
            "goal_features": self.goal_features.tolist() if len(self.goal_features) > 0 else [],
            "edges": self.edges,
            "metadata": self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def load(path: str) -> "GraphData":
        """Load graph data from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        graph = GraphData()
        graph.node_ids = data["node_ids"]
        graph.node_types = data["node_types"]
        graph.text_embeddings = np.array(data["text_embeddings"]) if data["text_embeddings"] else np.array([])
        graph.goal_features = np.array(data["goal_features"]) if data["goal_features"] else np.array([])
        graph.edges = [tuple(e) for e in data["edges"]]
        graph.metadata = data["metadata"]
        
        return graph


# ================================ GCN Retriever ================================

class GCNRetriever:
    """
    GCN-based retriever for memory graphs.
    Learns to embed nodes for similarity-based retrieval.
    """
    
    def __init__(
        self,
        working_dir: str,
        text_embed_dim: int = 384,
        goal_feature_dim: int = 25,
        hidden_dim: int = 128,
        output_dim: int = 64,
        model_type: str = "sage",
        device: str = "cpu"
    ):
        self.working_dir = working_dir
        self.device = device
        self.text_embed_dim = text_embed_dim
        self.goal_feature_dim = goal_feature_dim
        self.output_dim = output_dim
        
        # Graph data
        self.graph_data = GraphData()
        self.graph_path = os.path.join(working_dir, "gcn_graph_data.json")
        
        # Model
        self.model = None
        self.model_path = os.path.join(working_dir, "gcn_model.pt")
        
        # Training examples
        self.training_examples: List[RetrievalExample] = []
        
        # Node embeddings cache (updated after training)
        self.node_embeddings: Optional[np.ndarray] = None
        
        if HAS_TORCH:
            self._init_model(text_embed_dim, goal_feature_dim, hidden_dim, output_dim, model_type)
            self._load()
    
    def _init_model(self, text_embed_dim, goal_feature_dim, hidden_dim, output_dim, model_type):
        """Initialize the GCN model."""
        self.model = MemoryGraphEncoder(
            text_embed_dim=text_embed_dim,
            goal_feature_dim=goal_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            model_type=model_type
        ).to(self.device)
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        text_embedding: np.ndarray,
        goal_features: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a node to the graph."""
        idx = self.graph_data.add_node(
            node_id=node_id,
            node_type=node_type,
            text_embedding=text_embedding,
            goal_feature=goal_features,
            metadata=metadata
        )
        
        # Invalidate cache
        self.node_embeddings = None
        
        return idx
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "related"):
        """Add an edge between nodes."""
        self.graph_data.add_edge(source_id, target_id, edge_type)
        self.node_embeddings = None
    
    def add_training_example(
        self,
        query_text_embed: np.ndarray,
        query_goal_features: Optional[np.ndarray],
        retrieved_node_id: str,
        success: bool,
        negative_node_ids: Optional[List[str]] = None
    ):
        """
        Add a training example from a completed task.
        
        Args:
            query_text_embed: Text embedding of the query
            query_goal_features: Goal features of the query
            retrieved_node_id: ID of the node that was retrieved
            success: Whether the task succeeded
            negative_node_ids: IDs of irrelevant nodes (for contrastive learning)
        """
        positive_idx = self.graph_data.get_node_idx(retrieved_node_id)
        if positive_idx < 0:
            return
        
        negative_idxs = []
        if negative_node_ids:
            for neg_id in negative_node_ids:
                idx = self.graph_data.get_node_idx(neg_id)
                if idx >= 0:
                    negative_idxs.append(idx)
        
        example = RetrievalExample(
            query_text_embed=query_text_embed,
            query_goal_features=query_goal_features,
            positive_idx=positive_idx,
            negative_idxs=negative_idxs,
            label=1.0 if success else 0.0
        )
        
        self.training_examples.append(example)
    
    def train(self, epochs: int = 10, lr: float = 0.001, batch_size: int = 32):
        """
        Train the GCN model on collected examples.
        Uses contrastive learning with triplet loss.
        """
        if not HAS_TORCH or self.model is None:
            print("PyTorch not available, skipping training")
            return
        
        if len(self.training_examples) < 10:
            print(f"Not enough training examples ({len(self.training_examples)}), need at least 10")
            return
        
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=lr)
        
        # Prepare graph data
        text_features = torch.FloatTensor(self.graph_data.text_embeddings).to(self.device)
        goal_features = None
        if len(self.graph_data.goal_features) > 0:
            goal_features = torch.FloatTensor(self.graph_data.goal_features).to(self.device)
        
        edge_index = self.graph_data.get_edge_index()
        if edge_index is not None:
            edge_index = torch.LongTensor(edge_index).to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle examples
            np.random.shuffle(self.training_examples)
            
            for i in range(0, len(self.training_examples), batch_size):
                batch = self.training_examples[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Get node embeddings
                node_embeds = self.model(text_features, goal_features, edge_index)
                
                # Compute triplet loss for each example
                batch_loss = 0.0
                for ex in batch:
                    # Query embedding (use the positive node's embedding as anchor)
                    query_embed = torch.FloatTensor(ex.query_text_embed).to(self.device)
                    query_embed = self.model.text_proj(query_embed.unsqueeze(0))
                    query_embed = F.normalize(query_embed, p=2, dim=-1)
                    
                    # Positive embedding
                    pos_embed = node_embeds[ex.positive_idx].unsqueeze(0)
                    
                    # Positive similarity
                    pos_sim = F.cosine_similarity(query_embed, pos_embed)
                    
                    # Negative similarities (if available)
                    if ex.negative_idxs:
                        neg_embeds = node_embeds[ex.negative_idxs]
                        neg_sims = F.cosine_similarity(query_embed, neg_embeds)
                        max_neg_sim = neg_sims.max()
                        
                        # Triplet margin loss
                        margin = 0.2
                        loss = F.relu(max_neg_sim - pos_sim + margin)
                    else:
                        # Just maximize positive similarity
                        loss = 1.0 - pos_sim
                    
                    # Weight by success
                    loss = loss * (1.0 + ex.label)  # More weight on successful examples
                    batch_loss += loss
                
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
            
            avg_loss = total_loss / (len(self.training_examples) / batch_size)
            if (epoch + 1) % 5 == 0:
                print(f"GCN Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Update cached embeddings
        self._update_embeddings()
        self._save()
    
    def _update_embeddings(self):
        """Update cached node embeddings after training."""
        if not HAS_TORCH or self.model is None:
            return
        
        if len(self.graph_data.text_embeddings) == 0:
            return
        
        self.model.eval()
        with torch.no_grad():
            text_features = torch.FloatTensor(self.graph_data.text_embeddings).to(self.device)
            goal_features = None
            if len(self.graph_data.goal_features) > 0:
                goal_features = torch.FloatTensor(self.graph_data.goal_features).to(self.device)
            
            edge_index = self.graph_data.get_edge_index()
            if edge_index is not None:
                edge_index = torch.LongTensor(edge_index).to(self.device)
            
            embeddings = self.model(text_features, goal_features, edge_index)
            self.node_embeddings = embeddings.cpu().numpy()
    
    def retrieve(
        self,
        query_text_embed: np.ndarray,
        query_goal_features: Optional[np.ndarray] = None,
        top_k: int = 5,
        node_type_filter: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve similar nodes given a query.
        
        Args:
            query_text_embed: Text embedding of the query
            query_goal_features: Goal features of the query
            top_k: Number of results to return
            node_type_filter: Only return nodes of this type
        
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if len(self.graph_data.node_ids) == 0:
            return []
        
        # Use cached embeddings if available
        if self.node_embeddings is not None and HAS_TORCH:
            # Compute query embedding through the model
            self.model.eval()
            with torch.no_grad():
                query_embed = torch.FloatTensor(query_text_embed).to(self.device)
                query_embed = self.model.text_proj(query_embed.unsqueeze(0))
                query_embed = F.normalize(query_embed, p=2, dim=-1)
                query_embed = query_embed.cpu().numpy()
            
            # Compute similarities
            similarities = np.dot(self.node_embeddings, query_embed.T).flatten()
        else:
            # Fallback: direct text embedding similarity
            similarities = np.dot(
                self.graph_data.text_embeddings, 
                query_text_embed
            ) / (
                np.linalg.norm(self.graph_data.text_embeddings, axis=1) * 
                np.linalg.norm(query_text_embed) + 1e-8
            )
        
        # Filter by node type if specified
        if node_type_filter:
            mask = np.array([
                1.0 if t == node_type_filter else 0.0 
                for t in self.graph_data.node_types
            ])
            similarities = similarities * mask
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                results.append((self.graph_data.node_ids[idx], float(similarities[idx])))
        
        return results
    
    def _save(self):
        """Save model and graph data."""
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Save graph data
        self.graph_data.save(self.graph_path)
        
        # Save model
        if HAS_TORCH and self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)
        
        # Save embeddings
        if self.node_embeddings is not None:
            np.save(os.path.join(self.working_dir, "node_embeddings.npy"), self.node_embeddings)
    
    def _load(self):
        """Load model and graph data."""
        # Load graph data
        if os.path.exists(self.graph_path):
            self.graph_data = GraphData.load(self.graph_path)
        
        # Load model
        if HAS_TORCH and self.model is not None and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # Load embeddings
        embed_path = os.path.join(self.working_dir, "node_embeddings.npy")
        if os.path.exists(embed_path):
            self.node_embeddings = np.load(embed_path)


# ================================ Fallback Retriever ================================

class FallbackRetriever:
    """
    Simple embedding-based retriever when PyTorch/PyG is not available.
    Uses cosine similarity on text embeddings.
    """
    
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.node_ids: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
    
    def add_node(
        self,
        node_id: str,
        text_embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """Add a node."""
        if node_id in self.node_ids:
            return
        
        self.node_ids.append(node_id)
        self.embeddings.append(text_embedding)
        self.metadata.append(metadata or {})
    
    def retrieve(
        self,
        query_embed: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve similar nodes."""
        if not self.embeddings:
            return []
        
        embeddings = np.array(self.embeddings)
        similarities = np.dot(embeddings, query_embed) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embed) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.node_ids[i], float(similarities[i])) for i in top_indices]

