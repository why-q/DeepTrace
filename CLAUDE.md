# 项目通用规范

## 0. AI 工作核心准则

- **理解先行**：在动手编码前，**必须**先阅读我提供的相关文件 (`@file`)，并向我复述你对任务的理解。
- **主动提问**：当需求不明确时，**必须**主动向我提问，**严禁**猜测
- **质疑精神**：如果发现我的需求存在逻辑矛盾或有更优实现，**必须**提出来与我讨论。
- **代码复用**：在创建新函数前，**必须**先在代码库中搜索是否存在可复用的类似功能。

## 1. 仓库概览

- **概述**
    - 该仓库用于实现伪造视频的片段级溯源数据集以及方法。
- **架构**
    - `asset/dataset` 存放数据集相关文件
    - `pretrained` 存放预训练模型
    - `logs` 存放日志
    - `src` 存放主要代码
        - `src/tracedino` 存放 tracedino 子项目的代码
        - `src/deeptrace` 存放 deeptrace 数据集子项目的代码
- **环境管理**
    - 使用 `uv` 进行环境管理，如果需要添加新的依赖，请尝试使用 `uv add` 而不是 `uv pip install`
    - 虚拟环境位于 `src/.venv`，配置文件位于 `src/pyproject.toml`

## 2. 全局核心命令

- 在运行任何代码前，请使用 `cd ~/pychen/DeepTrace` 进入项目文件夹，并使用 `source src/.venv/bin/activate` 激活虚拟环境。
- 若缺失任何运行代码所需的数据，请主动向我请求。
- 目前 `asset` 文件夹被我添加到了 `.gitignore` 中，但我希望其中的 `asset/dataset` 也使用 Git 进行管理。当我使用 Git 时，请对这个文件夹进行必要的特殊处理。

## 3. 渐进式披露：按需加载子项目规范

- 如果需要从全局角度快速理解该项目的核心思想，请参考 `ref/SUMMARY.md`；如果需要细致理解，请参考 `ref/core.tex`
- 如果你在 `asset/dataset` 中工作，你需要阅读 `asset/dataset/DATASET_SUMMARY.txt`
- 如果你在 `src/tracedino` 中工作，你需要阅读 `src/tracedino/CLAUDE.md`

## 4. 全局工作流