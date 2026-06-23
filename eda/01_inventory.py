"""01_inventory.py — 데이터 인벤토리 + 라벨 분포 확인.
실행: python eda/01_inventory.py   (프로젝트 루트에서)
산출: outputs/01_inventory.txt
"""
import os, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "01_inventory.txt")
L = []
def log(s=""): L.append(str(s))

# raw labeled (utf-8! cp949 아님) — 피처 + PassOrFail + Reason 모두 보유
df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
log(f"labeled_data.csv  shape={df.shape}")
log(f"PassOrFail:\n{df['PassOrFail'].value_counts(dropna=False).to_string()}")
log(f"\n불량(N) Reason:\n{df[df['PassOrFail']=='N']['Reason'].value_counts(dropna=False).to_string()}")

# CN7 단일 부품군으로 좁힘 (= 단일 공정 depth)
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)]
log(f"\nCN7 rows={len(cn7)}  불량={int((cn7['PassOrFail']=='N').sum())}")
log("CN7 불량 Reason (= eval 타깃):")
log(cn7[cn7['PassOrFail']=='N']['Reason'].value_counts(dropna=False).to_string())

# 전처리/표준화된 보조 파일들
for f in ["moldset_unlabeled_cn7.csv", "supervised_label_cn7.csv", "moldset_labeled_cn7.csv"]:
    g = pd.read_csv(os.path.join(D, f), index_col=0)
    pof = g['PassOrFail'].value_counts(dropna=False).to_dict() if 'PassOrFail' in g.columns else "라벨없음"
    log(f"\n{f}  shape={g.shape}  PassOrFail={pof}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
