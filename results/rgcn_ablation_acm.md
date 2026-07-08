# RGCN Ablation — acm

| Condition | n | test Macro-F1 | test Accuracy | val Macro-F1 |
|-----------|---|--------------|--------------|-------------|
| RGCN only | 10 | 0.9098 ± 0.0035 | 0.9086 ± 0.0036 | 0.9317 ± 0.0052 |
| RGCN + PDGNN (manual metapath) | 10 | 0.9063 ± 0.0090 | 0.9049 ± 0.0093 | 0.9295 ± 0.0050 |
| RGCN + GTN-PDGNN (learned metapath) | 10 | 0.9077 ± 0.0028 | 0.9064 ± 0.0028 | 0.9359 ± 0.0061 |
