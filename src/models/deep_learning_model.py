"""
Deep Learning Module for Smartphone Price Prediction.

Constructs a PyTorch Multi-Layer Perceptron (FFNN), applying strict
ReLU non-linear activations and heavy Dropout constraints natively targeting
Machine Learning regression bounds.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from src.models.model_evaluator import BaseModelTrainer
from src.logger import get_logger

logger = get_logger(__name__)

class SmartphonePriceFFNN(nn.Module):
    """
    Computes rigorous non-linear relationships via heavily constrained
    Dense PyTorch Matrix operations. Maps feature inputs structurally to 1 Float.
    """
    
    def __init__(self, input_dim: int):
        super(SmartphonePriceFFNN, self).__init__()
        
        # Hidden Layer Architectures aggressively bounding dimensionality down
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        
        # Output prediction node -> Unbounded Regression Target (Price)
        self.output_node = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        # High-dropout mathematically injected bounding small-dataset overfitting tendencies
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes sequential matrix logic bounding activations symmetrically.
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.output_node(x)
        return x

class DeepLearningTrainer(BaseModelTrainer):
    """
    Engine executing structural gradient descent tracking via PyTorch topologies natively.
    """
    
    def __init__(self, experiment_name: str = "Smartphone_Price_Prediction"):
        super().__init__(experiment_name=experiment_name)
        
        # Cast isolated Phase 2 artifacts into computationally rigid PyTorch gradients
        self.X_train_tensor = torch.FloatTensor(self.X_train)
        self.X_test_tensor = torch.FloatTensor(self.X_test)
        
        # Scikit-learn outputs 1D boundaries natively, PyTorch loss computations explicitly require 2D mappings [Batch, Features]
        self.y_train_tensor = torch.FloatTensor(self.y_train).view(-1, 1)
        self.y_test_tensor = torch.FloatTensor(self.y_test).view(-1, 1)

    def train_network(self) -> dict:
        """
        Dynamically initializes FFNN, executing Adam optimizers natively mapping RMSE across 100 Epochs.
        """
        with mlflow.start_run(run_name="PyTorch_FFNN_100Epochs"):
            input_dim = self.X_train_tensor.shape[1]
            model = SmartphonePriceFFNN(input_dim=input_dim)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            epochs = 100
            logger.info(f"Initiating PyTorch FFNN Optimization sequence ({epochs} Epochs)...")
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward computation
                predictions = model(self.X_train_tensor)
                loss = criterion(predictions, self.y_train_tensor)
                
                # Backward calculus constraints
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {loss.item():.4f}")
                    
            # Lock structure gracefully to scoring state
            model.eval()
            with torch.no_grad():
                y_pred_tensor = model(self.X_test_tensor)
                
            # ReCast to computationally lightweight numpy blocks supporting Phase 3 Tracker methods
            y_pred_np = y_pred_tensor.numpy().flatten()
            y_test_np = self.y_test_tensor.numpy().flatten()
            
            metrics = self.evaluate_model(y_test_np, y_pred_np)
            
            mlflow.log_params({"epochs": epochs, "optimizer": "Adam", "learning_rate": 0.01})
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "model")
            
            return metrics

if __name__ == "__main__":
    logger.info("Initializing Phase 3 Deep Learning Engine.")
    trainer = DeepLearningTrainer()
    trainer.train_network()
    logger.info("PyTorch Matrix logic sequence concluded successfully.")
