#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Chạy training trong tmux, log ra file, không chết khi rớt SSH.
#
#   ./run_train.sh                              # config/multilabel.yaml
#   ./run_train.sh config/multilabel_smoke.yaml
#   ./run_train.sh config/multilabel.yaml -o train_batch_size=8 -o data_root=/data/nih
#
# Kèm theo, cửa sổ tmux thứ hai chạy `nvidia-smi` để theo dõi VRAM.
#   tmux attach -t cdm        gắn vào session
#   Ctrl-b d                  thoát ra, training vẫn chạy
#   tmux kill-session -t cdm  dừng hẳn
# ---------------------------------------------------------------------------
set -euo pipefail

SESSION="${CDM_SESSION:-cdm}"
CONFIG="${1:-config/multilabel.yaml}"
shift || true
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/train_${STAMP}.log"
mkdir -p "$LOG_DIR"

command -v tmux >/dev/null || { echo "Chưa cài tmux: sudo apt install tmux"; exit 1; }
[ -f "$REPO/$CONFIG" ] || [ -f "$CONFIG" ] || { echo "Không thấy config: $CONFIG"; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session tmux '$SESSION' đang chạy sẵn."
    echo "  Xem:  tmux attach -t $SESSION"
    echo "  Dừng: tmux kill-session -t $SESSION"
    exit 1
fi

# `script -q -c` giữ output có màu và unbuffered khi ghi qua tee
CMD="cd '$REPO' && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
     python train_multilabel.py --config '$CONFIG' $* 2>&1 | tee '$LOG'; \
     echo; echo '=== KẾT THÚC (exit=\$?) — log: $LOG ==='; exec bash"

tmux new-session -d -s "$SESSION" -n train "$CMD"
tmux new-window -t "$SESSION" -n gpu "watch -n 5 nvidia-smi; exec bash"
tmux select-window -t "$SESSION:train"

echo "Đã khởi động session tmux '$SESSION' trên GPU $GPU"
echo "  config : $CONFIG"
echo "  log    : $LOG"
echo
echo "  tmux attach -t $SESSION      # xem tiến độ (Ctrl-b d để thoát ra)"
echo "  tail -f $LOG                 # hoặc theo dõi log từ shell khác"
