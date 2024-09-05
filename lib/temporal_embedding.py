import torch
import pdb

class TemporalEmbedding(torch.nn.Module):

    minute_size = 60
    hour_size = 24
    weekday_size = 7
    day_size = 32
    month_size = 12
    year_size = 20

    # [y:2024, m:8, wd:2, d:14, h:14, min10]

    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        self.minute_embed = torch.nn.Embedding(
            self.minute_size, d_model)
        self.hour_embed = torch.nn.Embedding(
            self.hour_size, d_model)
        self.weekday_embed = torch.nn.Embedding(
            self.weekday_size, d_model)
        self.day_embed = torch.nn.Embedding(
            self.day_size, d_model)
        self.month_embed = torch.nn.Embedding(
            self.month_size, d_model)
        self.year_embed = torch.nn.Embedding(
            self.year_size, d_model)

    def forward(self, x):
        x = x.long()
        minute = self.minute_embed(x[:,:,5])
        hour = self.hour_embed(x[:,:,4])
        weekday = self.weekday_embed(x[:,:,2])
        day = self.day_embed(x[:,:,3])
        month = self.month_embed(x[:,:,1])
        year = self.year_embed(x[:,:,0] - 2024)
        return minute + hour + weekday + day + month + year
