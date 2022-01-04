from torch import nn
from torch.distributions import Normal


class PictchedEncoder(nn.Module):

    def __init__(self, z_dim=256, cnn_in_chn=2):
        super(PictchedEncoder, self).__init__()

        self.cnn_out_chn = 20
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_in_chn, self.cnn_out_chn, kernel_size=(4, 12),
                      stride=(2, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)))

        self.gru = nn.GRU(self.cnn_out_chn * 39,
                          1024, batch_first=True,
                          bidirectional=True)

        self.linear_mu = nn.Linear(1024 * 2, z_dim)
        self.linear_var = nn.Linear(1024 * 2, z_dim)
        self.z_dim = z_dim

    def forward(self, x):
        # x: (bs, 2, 153, 88)
        # x = self.conv1(x)
        bs = x.size(0)
        x = self.cnn(x)

        lgth = x.size(2)
        x = x.permute(0, 2, 1, 3).reshape(bs, lgth, -1)

        x = self.gru(x)[-1]

        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)

        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()

        return Normal(mu, var)


class FrameEncoder3x153x88(nn.Module):

    def __init__(self, z_dim=192):
        super(FrameEncoder3x153x88, self).__init__()
        self.pitched_enc = PictchedEncoder(z_dim=z_dim, cnn_in_chn=3)

    @property
    def z_dim(self):
        return self.pitched_enc.z_dim

    def forward(self, pitched):
        return self.pitched_enc(pitched)
