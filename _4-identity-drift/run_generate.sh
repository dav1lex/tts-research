#!/bin/bash
set -e
PROJECT="/home/davilex/tts-research/_4-identity-drift"
VENV="/home/davilex/tts-research/_3-prosody-transfer-benchmark/.venv/bin/python"
VENV_XTTS="/home/davilex/tts-research/_3-prosody-transfer-benchmark/.venv-xtts/bin/python"

echo "=== GENERATING CHATTERBOX ==="
$VENV "$PROJECT/scripts/generate_chatterbox.py"

echo "=== GENERATING XTTS ==="
$VENV_XTTS "$PROJECT/scripts/generate_xtts.py"

echo "=== GENERATING KOKORO ==="
$VENV "$PROJECT/scripts/generate_kokoro.py"

echo "=== ALL GENERATION COMPLETE ==="
