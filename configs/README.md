# Retired JSON configuration

The flat JSON overlay interface has been removed. Complete YAML projects are in
[`examples/`](../examples/), and `pelec-post init PROJECT_DIR` creates a new
strict project. Historical JSON files are retained under `legacy/configs/` only
for provenance and are deliberately rejected by the current CLI.
