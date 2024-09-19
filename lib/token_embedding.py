import torch

class TokenEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = torch.nn.Conv1d(
            in_channels=c_in, out_channels=d_model, 
            kernel_size=3, padding=padding, padding_mode='circular')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
