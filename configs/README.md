# Configuration profiles

`pelec_post.py` accepts multiple checked JSON overlays. They are applied from
left to right, so server-local paths can override a committed scientific case
without changing the analysis definition.

```bash
mkdir -p configs/local
cp configs/server.example.json configs/local/server.json
# Edit only paths and server-local locations in configs/local/server.json.

python3 pelec_post.py \
  --config configs/kernel_spectrum.json \
  --config configs/local/server.json \
  --validate-config

python3 pelec_post.py \
  --config configs/kernel_spectrum.json \
  --config configs/local/server.json \
  --write-effective-config effective_config.json
```

Files under `configs/local/` are ignored by Git. Scientific settings belong in
the committed case profile; absolute paths and server-specific archive
locations belong in the local profile.
