"""공개 사전학습 대형 모델 선별 — 고정(frozen) 다층 특징 + Ridge (학습 없음).

timm `forward_intermediates`로 마지막 N개 층(ViT 블록 / CNN stage)의 특징맵을 받아 채널별 평균·표준편차를 이어 붙이고,
D4 TTA 평균 후 고정 5-Fold Ridge로 OOF·test 예측을 만든다. 모델별 입력 해상도(업샘플)를 지정할 수 있다.

예) python src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --tta 4
산출물: outputs/probe2/<model>_<size>/{oof.csv, test_pred.csv}, outputs/probe2_results.csv
라이선스는 timm pretrained_cfg로 확인 후 사용 (Apache-2.0 / MIT / CC-BY-NC-4.0 → 대회 허용)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, d4, get_folds, load_split  # noqa: E402

VIT_LIKE = ("vit", "eva", "deit", "beit", "samvit")


class HFDinov2(torch.nn.Module):
    """transformers Dinov2Model(예: facebook/webssl-dino*)을 timm forward_intermediates 형식으로 감싼다:
    지정 블록 출력(최종 norm 전)의 패치 토큰을 B,C,h,w로 반환."""

    def __init__(self, repo):
        super().__init__()
        from transformers import AutoModel
        if repo.lower().startswith("intel/dpt"):
            # MiDaS v3 DPT (Intel, 2021 공개; dpt-large = ViT-L/16 백본을 깊이 추정으로 학습, Apache-2.0) → 백본(DPTModel)만 사용, 입력 정규화 0.5/0.5
            from transformers import DPTForDepthEstimation
            self.m = DPTForDepthEstimation.from_pretrained(repo, torch_dtype=torch.float32).dpt
        elif "depth-anything" in repo.lower():
            # Depth Anything V2 (2024-06 공개; Large는 CC-BY-NC-4.0): DINOv2 백본을 깊이 추정으로 다시 학습 → 백본(Dinov2Backbone)만 사용
            from transformers import AutoModelForDepthEstimation
            self.m = AutoModelForDepthEstimation.from_pretrained(repo, torch_dtype=torch.float32).backbone
        else:
            self.m = AutoModel.from_pretrained(repo, torch_dtype=torch.float32)
        self.p = self.m.config.patch_size
        self.nreg = getattr(self.m.config, "num_register_tokens", 0) or 0
        lic = {"facebook/webssl": "cc-by-nc-4.0", "kaiko-ai/midnight": "mit", "depth-anything/Depth-Anything-V2-": "cc-by-nc-4.0",
               "LiheYoung/depth-anything-large": "apache-2.0", "Intel/dpt-large": "apache-2.0"}
        # kaiko-ai/midnight (Midnight-12k: DINOv2-g를 공개 병리 슬라이드 TCGA로 자기지도 학습, MIT, 2025-03-25 공개)은 입력 정규화가 0.5/0.5
        half = repo.startswith("kaiko-ai/midnight") or repo.lower().startswith("intel/dpt")
        self.pretrained_cfg = {"mean": (0.5, 0.5, 0.5) if half else (0.485, 0.456, 0.406), "std": (0.5, 0.5, 0.5) if half else (0.229, 0.224, 0.225),
                               "license": next((v for k, v in lic.items() if repo.startswith(k)), None)}

    def forward_intermediates(self, x, indices, intermediates_only=True):
        n = len(self.m.encoder.layer)
        idx = list(range(n - indices, n)) if isinstance(indices, int) else list(indices)
        hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        h, w = x.shape[-2] // self.p, x.shape[-1] // self.p
        return [hs[i + 1][:, 1 + self.nreg:].transpose(1, 2).reshape(x.shape[0], -1, h, w) for i in idx]


class FeatOnly(torch.nn.Module):
    """timm features_only CNN의 한 단계 특징맵을 forward_intermediates 형식으로 감싼다.
    모델명 'cnn:<timm 이름>@<단계 번호>' (예: cnn:resnet50.a1_in1k@2 = layer2, stride 8). ImageNet 사전학습 CNN의 얕은 단계를
    고정 특징으로 쓰고 DINO와 같은 풀링(hetero3·상 조건부)을 적용한다."""

    def __init__(self, spec):
        super().__init__()
        name, idx = spec.split("@")
        self.m = timm.create_model(name, pretrained=True, features_only=True, out_indices=(int(idx),))
        self.pretrained_cfg = self.m.pretrained_cfg

    def forward_intermediates(self, x, indices=None, intermediates_only=True):
        return self.m(x)


class Capi(torch.nn.Module):
    """CAPI ViT-L/14 (facebookresearch/capi, 코드·가중치 Apache-2.0, 2025-02-13 공개)를 forward_intermediates 형식으로 감싼다.
    모델명 'capi_vitl14_{lvd,in22k,p205,in1k}'. 지정 블록 출력(최종 norm 전)의 패치 토큰을 B,C,h,w로 반환(레지스터 16개 제외).
    위치 부호가 RoPE(0~1 좌표 × 각주파수)라 bf16에서는 각도 정밀도가 부족 → 항상 fp32로 계산."""

    def __init__(self, name):
        super().__init__()
        self.m = torch.hub.load("facebookresearch/capi:main", name, trust_repo=True)
        self.pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "license": "apache-2.0"}

    def forward_intermediates(self, x, indices, intermediates_only=True):
        n = self.m.n_blocks
        idx = list(range(n - indices, n)) if isinstance(indices, int) else list(indices)
        with torch.autocast("cuda", enabled=False):
            t, coords, _ = self.m.prepare_tokens_and_drop(x.float(), None)
            hs = self.m.encoder(t, coords=coords, return_layers={i + 1 for i in idx})
        h, w = x.shape[-2] // self.m.patch_size, x.shape[-1] // self.m.patch_size
        return [hs[i + 1][:, self.m.num_prefix_tokens:].transpose(1, 2).reshape(x.shape[0], -1, h, w) for i in idx]


def build_moge(repo, size):
    """MoGe(Microsoft, MIT; DINOv2-L을 단안 3차원 형상(점 지도) 추정으로 재학습)의 백본만 timm DINOv2-L 구조에 올린다.
    'moge:Ruicheng/moge-vitl'(2024-10) / 'moge:Ruicheng/moge-2-vitl'(2025-06). 위치 임베딩은 입력 격자에 맞게 보간."""
    from huggingface_hub import hf_hub_download
    from timm.layers import resample_abs_pos_embed
    ck = torch.load(hf_hub_download(repo, "model.pt"), map_location="cpu", weights_only=False)["model"]
    pre = "encoder.backbone." if any(k.startswith("encoder.backbone.") for k in ck) else "backbone."
    sd = {k[len(pre):]: v for k, v in ck.items() if k.startswith(pre)}
    m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=size)
    g = size // 14
    sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=(g, g), num_prefix_tokens=1)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  [moge] {repo}: 미로드 {len(miss)} {miss[:3]} | 무시 {len(unexp)} {unexp[:3]}", flush=True)
    mk = "encoder.image_mean" if "encoder.image_mean" in ck else "image_mean"
    mean = ck[mk].float().view(1, 3, 1, 1).cuda() if mk in ck else torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1)
    sk = mk.replace("mean", "std")
    std = ck[sk].float().view(1, 3, 1, 1).cuda() if sk in ck else torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    m.pretrained_cfg = {**m.pretrained_cfg, "license": "mit"}
    return m.cuda().eval(), mean, std


def build(name, size):
    if name.startswith("moge:"):
        return build_moge(name[5:], size)
    if name.startswith("capi_"):
        m = Capi(name).cuda().eval()
        mean = torch.tensor(m.pretrained_cfg["mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(m.pretrained_cfg["std"], device="cuda").view(1, 3, 1, 1)
        return m, mean, std
    if name.startswith("cnn:"):
        m = FeatOnly(name[4:]).cuda().eval()
        mean = torch.tensor(m.pretrained_cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(m.pretrained_cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
        return m, mean, std
    if name.startswith("hf:"):
        m = HFDinov2(name[3:]).cuda().eval()
        mean = torch.tensor(m.pretrained_cfg["mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(m.pretrained_cfg["std"], device="cuda").view(1, 3, 1, 1)
        return m, mean, std
    kw = {}
    if name.startswith(VIT_LIKE):
        kw["img_size"] = size
    m = timm.create_model(name, pretrained=True, num_classes=0, **kw).cuda().eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
    return m, mean, std


_PROJ: dict = {}


def pool(z: torch.Tensor, mode: str) -> torch.Tensor:
    """특징맵(B,C,H,W) → 채널별 통계.
    meanstd: 평균·표준편차 / q: + 10·50·90% 분위수 / gram: + 그람(공분산) 행렬 상삼각 —
    채널을 고정 난수 직교투영으로 64차원으로 줄인 뒤 토큰 간 공분산 (질감 기술자, 정답 미사용)"""
    z = z.float().flatten(2)
    feats = [z.mean(2), z.std(2)]
    if mode == "gram":
        C = z.shape[1]
        if C not in _PROJ:
            g = torch.Generator().manual_seed(C)
            q, _ = torch.linalg.qr(torch.randn(C, 64, generator=g))
            _PROJ[C] = q.to(z.device)
        t = torch.einsum("bcn,ck->bnk", z, _PROJ[C])
        t = t - t.mean(1, keepdim=True)
        cov = torch.einsum("bnk,bnl->bkl", t, t) / t.shape[1]
        iu = torch.triu_indices(64, 64, device=z.device)
        feats.append(cov[:, iu[0], iu[1]])
    if mode == "hetero":  # 공간 불균일성: 2×2·4×4 영역 평균들의 영역 간 표준편차 (위치별 미세구조 차이)
        B, Cc, N = z.shape
        h = int(round(N ** 0.5))
        zz = z.view(B, Cc, h, h)
        for g in (2, 4):
            rm = torch.nn.functional.adaptive_avg_pool2d(zz, g).flatten(2)
            feats.append(rm.std(2))
    if mode == "hetero3":  # hetero + 8×8 영역 평균 불균일성 + 4×4 영역 내부 표준편차들의 영역 간 표준편차
        B, Cc, N = z.shape
        h = int(round(N ** 0.5))
        zz = z.view(B, Cc, h, h)
        for g in (2, 4, 8):
            feats.append(torch.nn.functional.adaptive_avg_pool2d(zz, g).flatten(2).std(2))
        m1 = torch.nn.functional.adaptive_avg_pool2d(zz, 4)
        m2 = torch.nn.functional.adaptive_avg_pool2d(zz * zz, 4)
        feats.append((m2 - m1 * m1).clamp_min(1e-12).sqrt().flatten(2).std(2))  # 1e-12: 미세조정 역전파 NaN 방지
    if mode == "hetac":  # hetero3 + 채널별 공간 자기상관(가로·세로 평균, 지연 1·2·4·8 토큰) = 질감 변화의 공간 척도
        B, Cc, N = z.shape
        h = int(round(N ** 0.5))
        zz = z.view(B, Cc, h, h)
        for g in (2, 4, 8):
            feats.append(torch.nn.functional.adaptive_avg_pool2d(zz, g).flatten(2).std(2))
        m1 = torch.nn.functional.adaptive_avg_pool2d(zz, 4)
        m2 = torch.nn.functional.adaptive_avg_pool2d(zz * zz, 4)
        feats.append((m2 - m1 * m1).clamp_min(0).sqrt().flatten(2).std(2))
        zc = zz - zz.mean((2, 3), keepdim=True)
        var = (zc * zc).mean((2, 3)) + 1e-6
        for d in (1, 2, 4, 8):
            ac = 0.5 * ((zc[:, :, d:, :] * zc[:, :, :-d, :]).mean((2, 3)) + (zc[:, :, :, d:] * zc[:, :, :, :-d]).mean((2, 3)))
            feats.append(ac / var)
    if mode == "hetero5":  # 영역평균 불균일성(2·4) + 영역내 표준편차의 영역 간 std (2·4·8 격자)
        B, Cc, N = z.shape
        h = int(round(N ** 0.5))
        zz = z.view(B, Cc, h, h)
        for g in (2, 4):
            feats.append(torch.nn.functional.adaptive_avg_pool2d(zz, g).flatten(2).std(2))
        for g in (2, 4, 8):
            m1 = torch.nn.functional.adaptive_avg_pool2d(zz, g)
            m2 = torch.nn.functional.adaptive_avg_pool2d(zz * zz, g)
            feats.append((m2 - m1 * m1).clamp_min(0).sqrt().flatten(2).std(2))
    if mode == "q":
        feats.append(torch.quantile(z, torch.tensor([0.1, 0.5, 0.9], device=z.device), dim=2).permute(1, 0, 2).flatten(1))
    return torch.cat(feats, 1)


PHASE_TH = 0.08  # --phase-th


CONTRIB_TH = (0.04, 0.08, 0.15, 0.25)  # phase2c 기여량의 어둡기 단계 (옅은 2상 / 2상 / 짙은 2상 / 기공)


def dark_maps(x: torch.Tensor, grid: int, ths=CONTRIB_TH) -> torch.Tensor:
    """어둡기 단계별 지도: 클로징(5×5) 후 기지 밝기(70% 분위) 대비 th 이상 어두운 화소 비율 → (B,len(ths),g,g)"""
    c = -F.max_pool2d(-F.max_pool2d(x, 5, 1, 2), 5, 1, 2)
    m = torch.quantile(c.flatten(1), 0.7, dim=1).view(-1, 1, 1, 1)
    rel = (m - c) / m
    return F.adaptive_avg_pool2d(torch.cat([(rel > t).float() for t in ths], 1), grid)


def dark_map(x: torch.Tensor, grid: int) -> torch.Tensor:
    """상 지도: 그레이 클로징(5×5, 얇은 결정립계 제거) 후 기지 밝기(이미지 70% 분위) 대비 8% 이상 어두운 화소 → 토큰 격자 평균 (B,1,g,g)"""
    c = -F.max_pool2d(-F.max_pool2d(x, 5, 1, 2), 5, 1, 2)
    m = torch.quantile(c.flatten(1), 0.7, dim=1).view(-1, 1, 1, 1)
    d = ((m - c) / m > PHASE_TH).float()
    return F.adaptive_avg_pool2d(d, grid)


def boundary_map(x: torch.Tensor, grid: int, th: float = 0.05) -> torch.Tensor:
    """결정립계 지도: 블랙 톱햇(5×5 클로징 − 원본 = 얇은 어두운 선)이 기지 밝기(70% 분위)의 th 이상인 화소 → 토큰 격자 평균 (B,1,g,g)"""
    c = -F.max_pool2d(-F.max_pool2d(x, 5, 1, 2), 5, 1, 2)
    m = torch.quantile(c.flatten(1), 0.7, dim=1).view(-1, 1, 1, 1)
    return F.adaptive_avg_pool2d(((c - x) / m > th).float(), grid)


def class_maps(x: torch.Tensor, grid: int, edges=(-0.04, 0.06, 0.3)) -> torch.Tensor:
    """상 클래스 지도: 클로징(5×5) 후 기지 밝기(70% 분위) 대비 상대 어둡기 rel → [밝은 결정립 | 기지 | 2상 | 기공] 4클래스
    → 토큰 격자별 클래스 비율 (B,4,g,g)"""
    c = -F.max_pool2d(-F.max_pool2d(x, 5, 1, 2), 5, 1, 2)
    m = torch.quantile(c.flatten(1), 0.7, dim=1).view(-1, 1, 1, 1)
    rel = (m - c) / m
    e = [-1e9, *edges, 1e9]
    maps = [((rel > lo) & (rel <= hi)).float() for lo, hi in zip(e[:-1], e[1:])]
    return F.adaptive_avg_pool2d(torch.cat(maps, 1), grid)


def pool_classes(z: torch.Tensor, cm: torch.Tensor) -> torch.Tensor:
    """클래스별 가중 평균·표준편차 (채널별): [평균×K, std×K]"""
    z = z.float().flatten(2)
    mus, sds = [], []
    for j in range(cm.shape[1]):
        ww = cm[:, j].flatten(1).unsqueeze(1)
        sw = ww.sum(2, keepdim=True).clamp_min(1e-3)
        mu = (z * ww).sum(2, keepdim=True) / sw
        mus.append(mu.squeeze(2))
        sds.append((((z - mu) ** 2 * ww).sum(2, keepdim=True) / sw).clamp_min(1e-12).sqrt().squeeze(2))
    return torch.cat(mus + sds, 1)


def pool_phase(z: torch.Tensor, dm: torch.Tensor) -> torch.Tensor:
    """상 조건부 풀링: 토큰을 2상 비율 d / 기지 비율 (1-d)로 가중한 평균·표준편차 (채널별) → [2상 평균, 기지 평균, 2상 std, 기지 std]"""
    z = z.float().flatten(2)  # B,C,N
    w = dm.flatten(1).unsqueeze(1)  # B,1,N
    out = []
    for ww in (w, 1 - w):
        s = ww.sum(2, keepdim=True).clamp_min(1e-3)
        mu = (z * ww).sum(2, keepdim=True) / s
        out.append(mu.squeeze(2))
        out.append((((z - mu) ** 2 * ww).sum(2, keepdim=True) / s).clamp_min(1e-12).sqrt().squeeze(2))
    return torch.cat([out[0], out[2], out[1], out[3]], 1)


def pool_weights(z: torch.Tensor, ws) -> torch.Tensor:
    """가중치 지도 목록별 가중 평균·표준편차 (채널별) → [평균들..., 표준편차들...]"""
    z = z.float().flatten(2)
    mus, sds = [], []
    for w in ws:
        w = w.flatten(1).unsqueeze(1)
        s = w.sum(2, keepdim=True).clamp_min(1e-3)
        mu = (z * w).sum(2, keepdim=True) / s
        mus.append(mu.squeeze(2))
        sds.append((((z - mu) ** 2 * w).sum(2, keepdim=True) / s).clamp_min(1e-12).sqrt().squeeze(2))
    return torch.cat(mus + sds, 1)


_RFF = {}


def pool_rff(z: torch.Tensor, dim: int = 2048, seed: int = 0) -> torch.Tensor:
    """토큰 분포의 커널 평균 임베딩: 토큰별 layer-norm → 고정 무작위 사영 + ReLU(아크코사인 커널 근사) → 토큰 평균 (B,dim).
    선형 평균 풀링이 토큰 평균만 보는 것과 달리 토큰 분포 전체(다봉성·꼬리)를 요약한다. 사영 행렬은 seed 고정."""
    z = z.float().flatten(2).transpose(1, 2)  # B,N,C
    C = z.shape[-1]
    if (C, dim, seed) not in _RFF:
        g = torch.Generator().manual_seed(seed)
        _RFF[(C, dim, seed)] = (torch.randn(C, dim, generator=g).div_(C ** 0.5).cuda(), torch.rand(dim, generator=g).mul_(2).sub_(1).cuda())
    W, b = _RFF[(C, dim, seed)]
    return torch.relu(F.layer_norm(z, (C,)) @ W + b).mean(1)


@torch.no_grad()
def embed(m, mean, std, X, size, tta, layers, bs, mode="meanstd", gain=1.0, interp="bilinear", shifts=1):
    out = []
    ks = list(range(8))[:tta] if tta in (1, 8) else [0, 1, 4, 5]
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        if gain != 1.0:  # 전 이미지 공통 고정 대비 증폭 (중심 0.57 = train 디노이즈 이미지 평균 밝기, 잘라내지 않음)
            xb = 0.57 + gain * (xb - 0.57)
        acc = 0
        # 격자 이동 TTA: 원본 4px(=448 입력에서 반 패치 7px) 이동 — D4는 패치 격자를 그대로 보존하므로 격자 정렬 잡음은 이동으로만 평균됨
        sh = [(0, 0), (4, 0), (0, 4), (4, 4)][:shifts]
        for k, (dy, dx) in [(k, s_) for k in ks for s_ in sh]:
            xk = d4(xb, k)
            if dy or dx:
                xk = F.pad(xk, (dx, 0, dy, 0), mode="reflect")[..., :xb.shape[-2], :xb.shape[-1]]
            xi = xk
            if size != xi.shape[-1]:
                xi = F.interpolate(xi, size=(size, size), mode=interp, align_corners=False)
            xi = (xi.repeat(1, 3, 1, 1) - mean) / std
            if mode != "phase2x":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
            if mode == "phase4":  # 4클래스 상 조건부 풀링 (+ hetero3): 레이어당 6 + 8 블록
                g = int(round(fs[0].shape[-1]))
                cm = class_maps(xk, g)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_classes(z, cm)], 1) for z in fs], 1)
            elif mode == "bnd2":  # 결정립계 조건부 풀링 (+ hetero3): 경계 근처 / 결정립 내부 가중 평균·std
                g = int(round(fs[0].shape[-1]))
                bm = boundary_map(xk, g)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, bm)], 1) for z in fs], 1)
            elif mode == "phase3":  # 2상 내부 / 상 경계면 / 기지 내부 3분할 풀링 (+ hetero3): 레이어당 6 + 6 블록
                g = int(round(fs[0].shape[-1]))
                dm = dark_map(xk, g)
                ws = (dm ** 2, 4 * dm * (1 - dm), (1 - dm) ** 2)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_weights(z, ws)], 1) for z in fs], 1)
            elif mode == "phase2x":  # phase2 + 혼합 법칙 기여량(2상 분율 × 2상 평균 = Σ d·z / N) + CLS 토큰: 레이어당 12 블록
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fp = m.forward_intermediates(xi, indices=layers, intermediates_only=True, return_prefix_tokens=True)
                g = int(round(fp[0][0].shape[-1]))
                dm = dark_map(xk, g)
                wd = dm.flatten(1).unsqueeze(1)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.float().flatten(2) * wd).mean(2),
                                                  pre[:, 0].float()], 1) for z, pre in fp], 1)
            elif mode == "phase2m":  # phase2 + 혼합 법칙 기여량(2상 분율 × 2상 평균 = Σ d·z / N): 레이어당 11 블록
                g = int(round(fs[0].shape[-1]))
                dm = dark_map(xk, g)
                wd = dm.flatten(1).unsqueeze(1)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.float().flatten(2) * wd).mean(2)], 1) for z in fs], 1)
            elif mode == "phase2c":  # phase2 + 어둡기 단계별 혼합 법칙 기여량(Σ d_t·z / N, 4단계): 레이어당 14 블록
                g = int(round(fs[0].shape[-1]))
                dm = dark_map(xk, g)
                ws = dark_maps(xk, g).flatten(2)  # B,T,N
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm),
                                                  torch.einsum("bcn,btn->btc", z.float().flatten(2), ws).flatten(1) / ws.shape[2]], 1) for z in fs], 1)
            elif mode == "phase2r":  # phase2 + 무작위 ReLU 특징 평균(커널 평균 임베딩, 2048차원 = 2블록): 레이어당 12 블록
                g = int(round(fs[0].shape[-1]))
                dm = dark_map(xk, g)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm), pool_rff(z)], 1) for z in fs], 1)
            elif mode == "phase2":  # 상 조건부 풀링 (+ hetero3)
                g = int(round(fs[0].shape[-1]))
                dm = dark_map(xk, g)
                acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm)], 1) for z in fs], 1)
            else:
                acc = acc + torch.cat([pool(z, mode) for z in fs], 1)
        out.append((acc / (len(ks) * len(sh))).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tta", type=int, default=4, choices=[1, 4, 8])
    ap.add_argument("--layers", type=int, nargs="+", default=[4],
                    help="정수 1개 = 마지막 N개 층, 여러 개 = 해당 층 번호 목록 (예: 2 5 8 11)")
    ap.add_argument("--idx", type=int, nargs="+", default=None,
                    help="정확한 층/단계 번호 목록 (1개여도 그 번호 그대로; CNN은 단계 번호 — 예: ResNet layer2 = 2). 지정하면 --layers 대신 사용")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--pool", choices=["meanstd", "q", "gram", "hetero", "hetero3", "hetero5", "hetac", "phase2", "phase4", "bnd2", "phase3", "phase2x", "phase2m", "phase2c", "phase2r"], default="meanstd")
    ap.add_argument("--bf16-weights", action="store_true", help="가중치를 bf16으로 (대형 모델 VRAM 절약)")
    ap.add_argument("--src", default=None, help="입력 이미지 캐시 npz (예: outputs/denoised/tv_c0.6.npz). 없으면 원본 PNG")
    ap.add_argument("--gain", type=float, default=1.0, help="고정 대비 배율 (모든 이미지 동일)")
    ap.add_argument("--interp", choices=["bilinear", "bicubic"], default="bilinear")
    ap.add_argument("--shifts", type=int, default=1, choices=[1, 2, 4], help="격자 이동 TTA 수 (원본 4px 이동)")
    ap.add_argument("--phase-th", type=float, default=0.08, help="phase2 풀링의 2상 판정 상대 어둡기 기준")
    a = ap.parse_args()
    global PHASE_TH
    PHASE_TH = a.phase_th
    torch.backends.cudnn.benchmark = True
    layers = a.layers[0] if len(a.layers) == 1 else a.layers
    if a.idx:
        layers, a.layers = list(a.idx), list(a.idx)
    ltag = ("" if a.layers == [4] else "_L" + "-".join(map(str, a.layers))) +         ("" if a.pool == "meanstd" else f"_P{a.pool}") + ("" if a.tta == 4 else f"_T{a.tta}") +         ("" if not a.src else "_S" + Path(a.src).stem) +         ("" if a.gain == 1.0 else f"_g{a.gain:g}") + ("" if a.interp == "bilinear" else "_bicubic") +         ("" if a.phase_th == 0.08 or a.pool != "phase2" else f"_th{a.phase_th:g}") + ("" if a.shifts == 1 else f"_sh{a.shifts}")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    if a.src:
        z = np.load(a.src)
        X, Xte = torch.from_numpy(z["train"]).unsqueeze(1), torch.from_numpy(z["test"]).unsqueeze(1)
    else:
        X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    t0 = time.time()
    m, mean, std = build(a.model, a.size)
    if a.bf16_weights:
        m = m.to(torch.bfloat16)
    lic = m.pretrained_cfg.get("license")
    Etr = embed(m, mean, std, X, a.size, a.tta, layers, a.bs, a.pool, a.gain, a.interp, a.shifts)
    Ete = embed(m, mean, std, Xte, a.size, a.tta, layers, a.bs, a.pool, a.gain, a.interp, a.shifts)
    sec = time.time() - t0
    oof, tp = np.zeros(len(y)), np.zeros(len(test))
    for k in range(5):
        tr, va = fold != k, fold == k
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 50))).fit(Etr[tr], y[tr])
        oof[va] = rg.predict(Etr[va])
        tp += rg.predict(Ete) / 5
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    tag = f"{a.model.replace('hf:', '').replace('cnn:', 'cnn_').replace(':', '_').replace('@', '_s').replace('/', '__')}_{a.size}{ltag}"
    d = OUT / "probe2" / tag
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "emb.npz", train=Etr.astype(np.float32), test=Ete.astype(np.float32))  # 결합·스태킹 재사용
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    res = OUT / "probe2_results.csv"
    rows = pd.read_csv(res).to_dict("records") if res.exists() else []
    rows = [x for x in rows if x["tag"] != tag] + [{"tag": tag, "model": a.model, "size": a.size, "license": lic,
                                                     "dim": Etr.shape[1], "oof_rmse": round(r, 3), "sec": round(sec)}]
    pd.DataFrame(rows).sort_values("oof_rmse").to_csv(res, index=False)
    print(f"[{tag}] license={lic} dim={Etr.shape[1]} probe OOF RMSE {r:.3f} ({sec:.0f}s)")


if __name__ == "__main__":
    main()
