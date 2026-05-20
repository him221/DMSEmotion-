# Open-source Model Files

This repository does not include large model files or datasets. They are distributed separately so the GitHub repository remains small and easy to clone.

## Recommended Distribution Strategy

Primary source:

- GitHub Releases: `https://github.com/<your-github-username>/DMSEmotion/releases/tag/v1.0-models`

Optional domestic mirror:

- Baidu Netdisk: `TODO: add your Baidu Netdisk link`
- Extraction code: `TODO`

As the repository owner, upload the following local files from:

```text
F:\FaceTest原版\DMSEmotion-release-models
```

to the GitHub Release named `v1.0-models`.

## Required Runtime Files

Users should download these three files and place them exactly as follows:

| Release asset | Destination inside the cloned repo | Purpose |
| --- | --- | --- |
| `PrivateTest_model.t7` | `Expression Recognition/FER2013_VGG19/PrivateTest_model.t7` | Facial expression VGG19 weights |
| `shape_predictor_68_face_landmarks.dat` | `Expression Recognition/shape_predictor_68_face_landmarks.dat` | dlib 68-point face landmark model |
| `model.safetensors` | `Chinese classification/results/checkpoint-86050/model.safetensors` | Fine-tuned text emotion classifier weights |

## SHA256 Checksums

Use these hashes to verify downloaded release assets:

| File | SHA256 |
| --- | --- |
| `PrivateTest_model.t7` | `E05650E0BE41B4D0BDBF371B2934B833FC3A6A5A9AD00F101C9D9B8E42FB0668` |
| `shape_predictor_68_face_landmarks.dat` | `FBDC2CB80EB9AA7A758672CBFDDA32BA6300EFE9B6E6C7A299FF7E736B11B92F` |
| `model.safetensors` | `B9A0B1AF9726E756F34275CE5168413C1086ED9658C517A3BC57EC9206F66BAC` |

## Files Kept in Git

The repository keeps small files needed by the code:

- `Chinese classification/results/checkpoint-86050/config.json`
- `Chinese classification/emotion_chinese_english/config.json`
- `Chinese classification/emotion_chinese_english/sentencepiece.bpe.model`
- `Chinese classification/emotion_chinese_english/tokenizer_config.json`
- `Chinese classification/emotion_chinese_english/special_tokens_map.json`
- `Chinese classification/bert-base-chinese/config.json`
- `Chinese classification/bert-base-chinese/vocab.txt`
- `Chinese classification/bert-base-chinese/tokenizer.json`
- `Expression Recognition/images/emojis/*.png`

## Why These Files Are Separate

- GitHub normal repositories reject files larger than 100 MB.
- Model weights and training checkpoints make cloning slow.
- Runtime records and datasets may contain private or license-sensitive data.
- Releases are better for downloadable binary assets.

## Deployment Steps for Users

```powershell
git clone https://github.com/<your-github-username>/DMSEmotion.git
cd DMSEmotion
pip install -r requirements.txt
```

Then download the three model files from the release page and place them in the paths listed above.

Run the main GUI:

```powershell
cd "Expression Recognition"
python app_gui2.py
```

Speech recognition is optional. To enable it:

```powershell
$env:XFYUN_APPID="your_appid"
$env:XFYUN_API_KEY="your_api_key"
$env:XFYUN_API_SECRET="your_api_secret"
python app_gui2.py
```
