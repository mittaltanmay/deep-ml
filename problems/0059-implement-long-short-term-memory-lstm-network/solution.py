import torch

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        hidden_all=[]
        for i in range(x.shape[0]):
            concatenated=torch.cat((initial_hidden_state,x[i].unsqueeze(1)), dim=0)
            ft=torch.sigmoid((self.Wf @ concatenated)+self.bf)
            it=torch.sigmoid((self.Wi @ concatenated)+self.bi)
            Ct=torch.tanh((self.Wc @ concatenated)+self.bc)
            ot=torch.sigmoid((self.Wo @ concatenated)+self.bo)
            initial_cell_state=(ft*initial_cell_state+Ct*it)
            initial_hidden_state=torch.tanh(initial_cell_state)*ot
            hidden_all.append(initial_hidden_state)
        hidden_all=torch.stack(hidden_all)
        return [hidden_all,initial_hidden_state,initial_cell_state]


        
