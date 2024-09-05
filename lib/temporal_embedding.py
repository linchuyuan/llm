import torch

class TemporalEmbedding(torch.nn.Module):

    minute_size = 60
    hour_size = 24
    weekday_size = 7
    day_size = 31
    month_size = 12
    year_size = 20

    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        self.minute_embed = torch.nn.Embed(minute_size, d_model)
        self.hour_embed = torch.nn.Embed(hour_size, d_model)
        self.weekday_embed = torch.nn.Embed(weekday_size, d_model)
        self.day_embed = torch.nn.Embed(day_size, d_model)
        self.month_embed = torch.nn.Embed(month_size, d_model)
        self.year_embed = torch.nn.Embed(year_size, d_model)

    def forward(self, x):
        minute = self.minute_embed(x[:,:,0])
        hour = self.hour_embed(x[:,:,1])
        weekday = self.weekday_embed(x[:,:,2])
        day = self.day_embed(x[:,:,3])
        month = self.month_embed(x[:,:,4])
        year = self.year_embed(x[:,:,5])
        return minute + hour + weekday + day + month + year
