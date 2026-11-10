# Reproduce train-only structure experiments. Run from the project root.
$ErrorActionPreference = 'Stop'
$env:HF_HUB_OFFLINE = '1' # Uses the already cached, Apache-2.0 timm weights.
$env:PYTHONIOENCODING = 'utf-8'
$py = '.venv/Scripts/python.exe'

& $py src/structure_features.py --split train --workers 4
if ($LASTEXITCODE -ne 0) { throw 'Feature extraction failed' }
& $py src/evaluate_structure.py --seeds 42 142 242
if ($LASTEXITCODE -ne 0) { throw 'Feature evaluation failed' }
foreach ($variant in @('raw', 'views', 'stats', 'fine_stats')) {
    & $py -u src/train_structure_cnn.py --variant $variant --resume
    if ($LASTEXITCODE -ne 0) { throw "CNN experiment failed: $variant" }
}
# After reviewing train-only results, inference is explicitly separate:
# & $py src/structure_features.py --split test --workers 4
# & $py src/evaluate_structure.py --predict-test fused
# & $py src/train_structure_cnn.py --variant SELECTED_VARIANT --predict-test
