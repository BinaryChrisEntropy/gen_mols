import torch


class Network(torch.nn.Module):
    def __init__(self, n_hidden=512, n_layers=2, batch_size=128, alpha_size=41):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.alpha_size = alpha_size

        self.drop = torch.nn.Dropout(p=0.2)
        self.lstm = torch.nn.LSTM(self.alpha_size, self.n_hidden,
                                  batch_first=True, dropout=0.2, num_layers=self.n_layers)
        self.decoder = torch.nn.Linear(self.n_hidden, self.alpha_size)

    def forward(self, x, prev_state):
        out, state = self.lstm(x, prev_state)
        out = self.drop(out)
        logits = self.decoder(out)
        return logits, state

    def init_state(self):
        return (torch.zeros(self.n_layers, self.batch_size, self.n_hidden),
                torch.zeros(self.n_layers, self.batch_size, self.n_hidden))