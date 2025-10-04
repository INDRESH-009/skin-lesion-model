# On-Device Skin Lesion Segmentation (WebGPU)

Lightweight segmentation model (**~350k params, ~3 MB**) running **fully in the browser** via **ONNX Runtime Web** (WebGPU → WebGL → WASM fallback). Data never leaves the device. PWA enabled for **offline** use.

## Try it
1. Open the deployed link (Vercel).
2. Select a lesion image (phone or desktop).
3. Adjust threshold. Download the binary mask if needed.

## Model
- Architecture: depthwise-separable UNet variant with LACM and boundary refinement (training only).
- Deploy variant: single-head logits, 1×1 output; input 256×256; ImageNet mean/std.
- Size: ~3 MB (FP16 ONNX).  
- Compute at 256²: ~0.6 GMACs (train config was ~2.5 GMACs at 512²).

## Pre/Post
- Pre: resize to 256×256, normalize by mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), NCHW.
- Post: sigmoid + threshold (default 0.5).

## Build/Deploy
Static site — push to GitHub and deploy via Vercel.  
`vercel.json` sets long cache for ONNX and immutable assets; HTML/SW are no-cache so updates apply on refresh.

## Offline (PWA)
First load caches app + model. Add to Home Screen to use offline.  
Note: The service worker version (`CACHE` in `sw.js`) should be bumped on each release.

## Safety
Research prototype, not a medical device. No diagnostic claims.

