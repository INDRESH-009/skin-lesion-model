import torch, onnx
from model import UNetDW512_BR_LACM  # your class

PT_PATH = "unetdw512_BR_LACM_best (1) (1).pt"   # your 3MB ckpt
ONNX_OUT = "lesion_256.onnx"
H = W = 256

device = "cpu"
net = UNetDW512_BR_LACM(out_channels=1, bottleneck_drop=0.1, return_boundary=False).eval()

ckpt = torch.load(PT_PATH, map_location=device)
sd = ckpt.get("model", ckpt)  # support both plain sd and {"model":sd}
# If EMA present, prefer it for floating tensors
if "ema" in ckpt:
    for k, v in sd.items():
        if k in ckpt["ema"] and torch.is_floating_point(sd[k]):
            sd[k] = ckpt["ema"][k]
net.load_state_dict(sd, strict=True)

dummy = torch.randn(1, 3, H, W, device=device)

torch.onnx.export(
    net, dummy, ONNX_OUT,
    input_names=["input"], output_names=["logits"],
    opset_version=17, do_constant_folding=True,
    dynamic_axes=None  # fixed size = simpler & faster in browser
)

onnx.checker.check_model(onnx.load(ONNX_OUT))
print("Saved:", ONNX_OUT)
