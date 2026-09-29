import torch
import torch.nn as nn


class Net(nn.Module):
    """Tabular MLP + dilated 1-D CNN over the last SEQ_LEN minutes, multi-task heads.

    Outputs: logits for Up@5m, Up@15m and auxiliary vol-normalised return regressions.
    """

    def __init__(self, n_feat, n_ch=3, hid=128, drop=0.2):
        super().__init__()
        self.tab = nn.Sequential(nn.Linear(n_feat, hid), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hid, hid), nn.GELU())
        self.cnn = nn.Sequential(
            nn.Conv1d(n_ch, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid + 64, hid), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hid, 4))

    def forward(self, x, s):
        h = self.cnn(s.transpose(1, 2))
        z = torch.cat([self.tab(x), h.mean(-1), h[..., -1]], dim=1)
        return self.head(z)
