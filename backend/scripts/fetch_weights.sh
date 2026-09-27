#!/bin/sh
# Pre-fetch PANNs checkpoints with curl + sha256 verify (F13).
# panns_inference auto-downloads via `os.system('wget ...')`; macOS has no wget and the
# failure is silent, so every tagging stage dies on a missing file. Both classes are
# used: AudioTagging -> Cnn14_mAP=0.431.pth, SoundEventDetection -> Cnn14_DecisionLevelMax.pth.
set -eu
DIR="$HOME/panns_data"
mkdir -p "$DIR"

fetch() {  # fetch <dest-name> <url> <sha256>
  dest="$DIR/$1"
  if [ -f "$dest" ] && [ "$(shasum -a 256 "$dest" | cut -d' ' -f1)" = "$3" ]; then
    echo "ok       $1"; return
  fi
  echo "fetching $1"
  curl -fL --retry 3 -o "$dest.part" "$2"
  got=$(shasum -a 256 "$dest.part" | cut -d' ' -f1)
  if [ "$got" != "$3" ]; then
    echo "checksum mismatch for $1: $got" >&2; rm -f "$dest.part"; exit 1
  fi
  mv "$dest.part" "$dest"
}

fetch class_labels_indices.csv \
  "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" \
  cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429
fetch "Cnn14_mAP=0.431.pth" \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" \
  0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31
fetch Cnn14_DecisionLevelMax.pth \
  "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1" \
  dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142
