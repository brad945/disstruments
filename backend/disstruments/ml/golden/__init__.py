"""Golden set (ML_ENGINEERING §3.3): real songs Bradley knows well, hand-labelled by ear at
leaf level. Evaluation only; never trained on. Audio stays local in `~/datasets/golden/audio/`
and is never committed. Only the label JSONs (song metadata + instruments) are kept.

    python -m disstruments.ml.golden page      # writes ~/datasets/golden/label.html

Label semantics per leaf: present -> positive; unsure -> unknown (masked); absent ->
negative only if the labeller confirmed they listened to the whole song, else unknown.
"""
