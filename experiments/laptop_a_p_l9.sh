#!/usr/bin/env bash
# [노트북 10/18] DINOv2 4보기(U 경로) 특징 재추출(USB 선별본에서 빠짐) → A(dnpp0.25)와 측정 블록 P(mr_v14_pp_sc)에도 층 9 추가
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
D=vit_large_patch14_dinov2.lvd142m; L="9 11 13 15 17 19 21"; B=unet_cm7_am22_l1_i10000_boot; AR=0.01
dg() { echo "--group emb:$1:1:$2:1024:10:0,1,2,3,4,5 --group emb:$1:0.25:$2:1024:10:6,7,8,9"; }
for s in 48 32 64 96; do
  E4=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2_S${B}_fflowpass${s}_low
  [ -f $E4/emb.npz ] || $PY -W ignore src/probe_models.py --model $D --size 448 --layers $L --pool phase2 --bs 8 --src outputs/denoised/${B}_fflowpass${s}_low.npz 2>&1 | grep -E "probe OOF|Error|OOM"
  for spec in "1,2,3,4:chk" "0,1,2,3,4:L01234"; do LI=${spec%%:*}; NM=${spec##*:}
    n=dnpp0.25_${NM}_ff${s}_a$AR; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name $n $(dg $E4 $LI) | tail -1
  done
  if [ $s = 48 ]; then S48=${B}_fflowpass48_low
    for spec in "1,2,3,4:chk" "0,1,2,3,4:L01234"; do LI=${spec%%:*}; NM=${spec##*:}
      n=mr_v14_pp_sc_${NM}_a$AR; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name $n $(dg $E4 $LI) \
        --group "csv:outputs/features_cache/phase_$S48:0.381:pf_" --group "csv:outputs/features_cache/$S48:0.019:ph_" --group "csv:outputs/features_cache/$S48:0.15:(pore|gr|sp)_" --group "csv:outputs/features_cache/smcdf3_$S48:0.2:sc" | tail -1
    done
  fi
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
