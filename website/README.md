# Scene-Designer

**Learning Design Intelligence for Agentic 3D Scene Generation**

[Project page](https://11penguins.github.io/Scene-Designer/)

Xuanyu Liu*, Hongyan Feng*, Miao Pan, Sunlai Chen, Lizhou Liang, Bing Sun, Wenqi Zhang, Xuhong Zhang.

Zhejiang University; Universal Identification Technology (Hangzhou) Co., Ltd.

*Equal contribution. Contact: 22651306@zju.edu.cn

This repository contains the Scene-Designer code and project website.

## Code and models

See [code/README.md](../README.md) for trajectory compression and CoT annotation, Scene-OTC construction, SFT, Scene-GRPO, and five-view scene-generation inference. SceneSmith is prepared separately using the documented integration patch. See [validation notes](../VALIDATION.md) for executed checks and remaining verification limits.

- [Scene-Builder-Qwen3-VL-8B-Instruct](https://huggingface.co/XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct)
- [Scene-OTC-Bench](https://huggingface.co/datasets/XuanyuLiu/Scene-OTC-Bench)

The website includes the method, English overview video, prompt/video demos, and BibTeX citation.

## Local preview

```sh
python -m http.server 8891 --bind 127.0.0.1
```

Open http://127.0.0.1:8891/ on the same machine.

## Publishing

In repository Settings → Pages, use a GitHub Actions workflow to publish the `website/` directory. The site uses relative asset paths and requires no build step.

Edit index.html, styles.css, or app.js to update the page. Media files are in assets/; the Paper button is pending an arXiv URL. No manuscript PDF is included.

## Figure viewing

Figures 2–5 are exported as SVG from the paper figure sources, preserving vector text and original embedded images. Click any paper figure to inspect it at 100–300% zoom or open the original image. Hover motion respects the reduced-motion preference.

## Evaluation

The Evaluation section precedes the scene demos. Figure 29 presents qualitative comparisons, with a lightweight WebP preview and a zoomable SVG exported from the paper figure. The Scene-OTC results table reports pairwise comparison (94 examples) and full four-scene ranking (55 examples), with rounded accuracies transcribed from the paper.
