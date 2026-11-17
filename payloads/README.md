# Payloads

Prebuilt wire-format request files, uploaded verbatim to the client
pod's `/store/payloads/` (building them on the pod would put generator
noise inside the measured window).

Naming: `j0-*` JSON bodies, `triton-binary-*` KServe v2 binary frames
(header length goes in the `Inference-Header-Content-Length` request
header), `g-*` gRPC protobufs. See `docs/MODELS.md`.
