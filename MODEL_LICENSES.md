# Third-party model licenses

This repository’s source code is dual-licensed under **MIT OR Apache-2.0**
(see [`LICENSE`](LICENSE), [`LICENSE-MIT`](LICENSE-MIT), and
[`LICENSE-APACHE`](LICENSE-APACHE)).

**Model weights are not part of that dual license.** Weights are downloaded at
runtime (for example via `loco download`) from third-party providers. Use of
those models is governed by **each provider’s own terms**, which apply in
addition to this project’s code license.

## Models currently used by loco-bot

| Role | Hugging Face id | License (as published on Hub) | Notes / official link |
|------|-----------------|-------------------------------|------------------------|
| Chat (LiteRT-LM pack) | [`litert-community/gemma-4-E4B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm) | Apache-2.0 | Packaged from Gemma 4 E4B instruct |
| Chat (base card) | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | Apache-2.0 | [Gemma 4 license docs](https://ai.google.dev/gemma/docs/gemma_4_license) |
| S1 embedding | [`hotchpotch/bekko-embedding-v1-a8m`](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m) | MIT | ONNX + tokenizer used for topic detection |

Previously evaluated (not the current default S1 embedder):

| Role | Hugging Face id | License (as published on Hub) |
|------|-----------------|-------------------------------|
| S1 embedding (superseded) | [`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | Apache-2.0 |

## Disclaimer

The license names, links, and summaries in this file are provided for
convenience only and may become outdated as upstream cards or terms change.

**You are responsible for verifying the current license and acceptable-use
terms with each model provider before downloading, using, redistributing, or
building on any model weights.** The authors and contributors of loco-bot
accept no liability for reliance on information in this document that is
incomplete, inaccurate, or no longer current.
