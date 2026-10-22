"""DRUNet(gray) 네트워크 정의 — KAIR(MIT)의 UNetRes와 같은 구조(가중치 drunet_gray.pth를 strict 로드)."""
import torch, torch.nn as nn
class Res(nn.Module):
    def __init__(s, c): super().__init__(); s.res = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1, bias=False))
    def forward(s, x): return x + s.res(x)
class DRUNet(nn.Module):
    def __init__(s, nc=(64, 128, 256, 512), nb=4):
        super().__init__(); s.m_head = nn.Conv2d(2, nc[0], 3, 1, 1, bias=False)
        s.m_down1 = nn.Sequential(*[Res(nc[0]) for _ in range(nb)], nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False))
        s.m_down2 = nn.Sequential(*[Res(nc[1]) for _ in range(nb)], nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False))
        s.m_down3 = nn.Sequential(*[Res(nc[2]) for _ in range(nb)], nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False))
        s.m_body = nn.Sequential(*[Res(nc[3]) for _ in range(nb)])
        s.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=False), *[Res(nc[2]) for _ in range(nb)])
        s.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=False), *[Res(nc[1]) for _ in range(nb)])
        s.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=False), *[Res(nc[0]) for _ in range(nb)])
        s.m_tail = nn.Conv2d(nc[0], 1, 3, 1, 1, bias=False)
    def forward(s, x0):
        x1 = s.m_head(x0); x2 = s.m_down1(x1); x3 = s.m_down2(x2); x4 = s.m_down3(x3); x = s.m_body(x4)
        x = s.m_up3(x + x4); x = s.m_up2(x + x3); x = s.m_up1(x + x2); return s.m_tail(x + x1)
def load(path, device="cuda"):
    m = DRUNet(); m.load_state_dict(torch.load(path, map_location="cpu"), strict=True); return m.to(device).eval()
