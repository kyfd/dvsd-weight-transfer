Temporary private transport for publicly released research checkpoints.

Reassemble CountSE on Linux:

```bash
cat countse/countse_model_only.pth.part* > countse_model_only.pth
cat bert-base-uncased/model.safetensors.part* > bert-base-uncased/model.safetensors
cat clip/ViT-B-16.pt.part* > clip/ViT-B-16.pt
cat fsc147/FSC147_384_V2.zip.part* > fsc147/FSC147_384_V2.zip
```

This repository is deleted after transfer and SHA-256 verification.

Expected SHA-256 values:

- `countse_model_only.pth`: `581bdcfa77b0ca81d1464029ed41ca93d5a51a05ba7860ba923e81509d7f30b5`
- `lgcount_alignment.ckpt`: `38e9d4a8969d17a4f7bfe221afae1b6fd417417e7e96052e4239fdac0f2d79f0`
- `bert-base-uncased/model.safetensors`: `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`
- `clip/ViT-B-16.pt`: `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`
- `fsc147/FSC147_384_V2.zip`: `4f9eff24c39f956d614abdb8888f9f7ae84cce92f81842e4159de45cd54d965f`
