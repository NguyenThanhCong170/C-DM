"""
Gói đánh giá: TRTR / TSTR / CAS.

CỐ Ý ĐỂ RỖNG. Lần trước file này re-export từ .metrics/.splits/.classifier,
nên bất kỳ `import evaluation.<gì đó>` nào cũng kéo theo torch + torchvision +
sklearn, và một file bị đổi chỗ là cả gói sập với ModuleNotFoundError. Cứ import
thẳng module cần dùng: `from evaluation.metrics import evaluate`.
"""
