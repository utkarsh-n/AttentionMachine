# Public source snapshot

This repository contains a reviewed source snapshot. Original Git history, credentials, personal records, local signing identities, device diagnostics, build outputs, downloaded model weights, firmware, and unreviewed media are excluded. Existing upstream license and attribution files apply to their respective components. Publication does not grant a new license to third-party content.

Configure your own Apple signing team to run native apps on a physical device. Downloaded datasets, binary runtimes, fonts, demo media, and models may need to be obtained separately using the documented upstream sources or fetch/build tools. This publication review does not certify that this snapshot builds or runs.

## Project setup and omitted assets

Clone [OfflineIntelligenceKit](https://github.com/utkarsh-n/OfflineIntelligenceKit) beside this repository; `project.yml` references `../OfflineIntelligenceKit` and `../OfflineIntelligenceKit/MLX`. The shared text assistant's 2B/4B models are omitted from `Resources/Models/`; obtain the versions documented by that package and provide them locally.

The measured-attention exhibit additionally needs Qwen3 4B Instruct 2507. Its multi-gigabyte weights/tokenizers are omitted from `Resources/Model/`. `Runtime/model-manifest.json` records the pinned upstream revision and file checksums; `Runtime/setup.sh` and `Runtime/capture_trace.py` preserve the download and scientific trace pipeline. They are optional setup tools and have not been executed for publication. Review `Documentation/MODEL.md` for model licensing. The retained `Resources/RecordedTrace.json` is a fixed scientific demonstration fixture, not private chat history.

Create `Resources/Model/` and `Resources/Models/` after supplying the required assets, then regenerate the Xcode project with XcodeGen. Private installation, audio and device diagnostic reports have been removed.
