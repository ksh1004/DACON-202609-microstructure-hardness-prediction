# 외부 코드·가중치 고지

이 저장소의 코드는 [MIT 라이선스](LICENSE)입니다. 아래 파일은 외부 프로젝트의 코드나 가중치를 바탕으로 하며, 해당 원본의 라이선스 고지를 따릅니다.

| 대상 | 원본 | 라이선스 |
|---|---|---|
| `src/drunet_net.py` (DRUNet 구조 재구현), `src/drunet_finetune.py`·`src/drunet_denoise.py`가 쓰는 사전학습 가중치 `drunet_gray.pth` | [KAIR](https://github.com/cszn/KAIR) | MIT |
| 공개 가중치 `drunetft_ft1b.pt` (DRUNet gray를 미세조정한 것, [Releases](https://github.com/ksh1004/DACON-202609-microstructure-hardness-prediction/releases/tag/weights-v1)) | [KAIR](https://github.com/cszn/KAIR) | MIT |
| `src/scunet_net.py` (원본 `models/network_scunet.py`에서 프로파일링용 `thop` import와 print만 제거) | [SCUNet](https://github.com/cszn/SCUNet) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

특징 추출에 쓰는 사전학습 모델(DINOv2, Depth-Anything-V2, TIPSv2, timm ImageNet CNN)은 저장소에 포함하지 않고 실행할 때 각 배포처에서 내려받습니다.
라이선스는 [README 11장](README.md#11-공개-범위)에 정리했습니다. 특히 Depth-Anything-V2-Large는 CC-BY-NC-4.0(비상업적 이용만 허용)입니다.

---

## KAIR — MIT License

```
MIT License

Copyright (c) 2019 Kai Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
