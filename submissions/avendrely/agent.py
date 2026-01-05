"""
Agent submission for Simple Tag competition.
Loads the trained DQN model from rendu.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class QNetwork(nn.Module):
    """Q-Network matching the architecture from rendu.py"""
    def __init__(self, input_dim=14, hidden_dim=512, output_dim=5):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class StudentAgent:
    """
    Predator agent using trained DQN model.
    """
    
    def __init__(self):
        """
        Initialize your predator agent by loading the trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model
        self.submission_dir = Path(__file__).parent
        model_path = self.submission_dir / "predator_model.pth"
        
        self.model = QNetwork().to(self.device)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def get_action(self, observation, agent_id: str):
        """
        Get action using the trained Q-network.
        
        Args:
            observation: Agent's observation from the environment (numpy array, shape (14,))
            agent_id (str): Unique identifier for this agent instance
            
        Returns:
            action: Discrete action in range [0, 4]
        """
        # Convert observation to tensor
        obs = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get Q-values and select best action
        with torch.no_grad():
            q_values = self.model(obs)
            action = q_values.argmax().item()
        
        return action


if __name__ == "__main__":
    # Test the agent
    print("Testing StudentAgent...")
    
    # Test predator agent (adversary has 14-dim observation)
    predator_agent = StudentAgent()
    predator_obs = np.random.randn(14)  # Predator observation size
    predator_action = predator_agent.get_action(predator_obs, "adversary_0")
    print(f"Predator observation shape: {predator_obs.shape}")
    print(f"Predator action: {predator_action} (should be in [0, 4])")
    
    print("✓ Agent is working!")
