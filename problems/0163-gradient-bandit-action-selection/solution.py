import torch

class GradientBandit:
    def __init__(self, num_actions: int, alpha: float = 0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = torch.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0
        self.prob = torch.softmax(self.preferences, dim=0)
        self.action=-1
    
    def softmax(self) -> torch.Tensor:
        # Compute softmax probabilities from preferences
        self.prob=torch.softmax(self.preferences,dim=0)
        return self.prob
    def select_action(self) -> int:
        # Sample an action according to the softmax distribution
        self.action=torch.multinomial(self.prob,1).item()
        return self.action
    
    def update(self, action: int, reward: float):
        self.prob = torch.softmax(self.preferences, dim=0)
        # baseline=self.avg_reward
        self.time+=1
        self.avg_reward+=((reward-self.avg_reward)/self.time)
        indicator = torch.zeros(self.num_actions)
        indicator[action] = 1
        self.preferences+=self.alpha*(reward-self.avg_reward)*(indicator-self.prob)

