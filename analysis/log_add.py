"""submission_log.csv에 행 추가 (같은 파일명이 있으면 교체). 사용: python log_add.py <date> <file> <oof> <public|''> <models> <notes>"""
import sys, pandas as pd
p = "submissions/submission_log.csv"
d = pd.read_csv(p, dtype=str, keep_default_na=False)
date, f, oof, pub, models, notes = sys.argv[1:7]
pub = "" if pub == "-" else pub  # PowerShell 5.1은 빈 인자를 버리므로 '-'로 표기
d = d[d.file != f].reset_index(drop=True)
d.loc[len(d)] = [date, f, oof, pub, models, notes]
d.to_csv(p, index=False, encoding="utf-8")
print(d.tail(3).to_string())
