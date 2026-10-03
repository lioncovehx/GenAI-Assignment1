#!/bin/sh
# Downloads the seven trained ONNX models from the GitHub release into models/ and checks them against SHA256SUMS.
# Files that are already present and correct are skipped.
set -e
URL=https://github.com/Er0sama/GenAI-Assignment1/releases/download/models-v1
cd "$(dirname "$0")/models"
for f in $(awk '{print $2}' SHA256SUMS); do
  if [ -f "$f" ] && grep " $f\$" SHA256SUMS | sha256sum -c --status - 2>/dev/null; then continue; fi
  echo "downloading $f"
  curl -fL --retry 3 -o "$f" "$URL/$f"
done
sha256sum -c SHA256SUMS
