from torch import nn
from torch.distributions import Normal


class TextureEncoder(nn.Module):

    def __init__(self, emb_size=256, hidden_dim=1024, z_dim=256,
                 num_channel=10, return_h=False):
        super(TextureEncoder, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channel, kernel_size=(4, 12),
                                           stride=(4, 1), padding=0),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=(1, 4),
                                              stride=(1, 4)))
        self.fc1 = nn.Linear(num_channel * 29, 1000)
        self.fc2 = nn.Linear(1000, emb_size)
        self.gru = nn.GRU(emb_size, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.return_h = return_h

    def forward(self, pr):
        # pr: (bs, 32, 128)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).permute(0, 2, 1, 3).reshape(bs, 8, -1)
        pr_feat = self.fc2(self.fc1(pr))  # (bs, 8, emb_size)

        # hs, pr = self.gru(pr)
        pr = self.gru(pr_feat)[-1]

        pr = pr.transpose_(0, 1).contiguous()
        pr = pr.view(pr.size(0), -1)

        mu = self.linear_mu(pr)
        var = self.linear_var(pr).exp_()

        dist = Normal(mu, var)

        if self.return_h:
            return dist, pr_feat
        else:
            return dist
