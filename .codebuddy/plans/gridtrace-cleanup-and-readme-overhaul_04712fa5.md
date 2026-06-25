---
name: gridtrace-cleanup-and-readme-overhaul
overview: 删除 GitHub 远程仓库 8 个 dependabot 分支（保留 main），重构项目结构（清理冗余、重命名、补充缺失文件），并将 README 重写为 ≤100 行极简版本、置入封面图。
todos:
  - id: delete-remote-branches
    content: 使用 git push origin --delete 逐个删除 8 个 dependabot 远程分支，验证仅 main 保留
    status: completed
  - id: rename-gridtrace-pkg
    content: git mv gridtrace/ → src/gridtrace/，更新 pyproject.toml packages 配置
    status: completed
    dependencies:
      - delete-remote-branches
  - id: update-imports
    content: 扫描并更新 api/、tests/、scripts/ 中所有 from gridtrace.xxx import 语句为 from gridtrace.xxx
    status: completed
    dependencies:
      - rename-gridtrace-pkg
  - id: add-config-files
    content: 创建 .github/workflows/ci.yml、.gitignore、.env.example 三个缺失配置文件
    status: completed
    dependencies:
      - rename-gridtrace-pkg
  - id: cleanup-redundant-docs
    content: 删除冗余的 docs/api_reference.md 文件
    status: completed
  - id: rewrite-readme
    content: 重写 README.md 至 ≤100 行：加封面图 + 标题副标题、替换
    status: completed
    dependencies:
      - rename-gridtrace-pkg
      - add-config-files
      - cleanup-redundant-docs
---

## 任务概述

清理 GridTrace 项目的 GitHub 远程仓库（仅保留 main 分支），重构项目目录结构，添加缺失的配置文件，重写 README 为简洁版本并添加封面图。

## 详细需求

### 1. 删除远程非 main 分支

- 远程仓库：`https://github.com/guts-yang/GridTrace.git`
- 需删除 8 个 dependabot 分支：
- `dependabot/docker/docker/nginx-1.31-alpine`
- `dependabot/docker/docker/node-26-alpine`
- `dependabot/docker/docker/python-3.14-slim`
- `dependabot/github_actions/codecov/codecov-action-7`
- `dependabot/github_actions/docker/build-push-action-7`
- `dependabot/github_actions/docker/login-action-4`
- `dependabot/github_actions/docker/setup-buildx-action-4`
- `dependabot/github_actions/docker/setup-qemu-action-4`
- 保留 `main` 分支

### 2. 优化项目结构

- **清理冗余**：删除 `docs/api_reference.md`（README 精简后不再需要）
- **重命名**：将 `gridtrace/` → `src/gridtrace/`，使核心包更清晰，符合现代 Python 项目惯例
- **添加缺失文件**：
- `.github/workflows/ci.yml`（CI 徽章引用但不存在）
- `.gitignore`（顶层缺失）
- `.env.example`（README 引用但不存在）

### 3. 更新 README

- **精简至 ≤ 100 行**：仅保留 Overview + Algorithm（关键 1+2+3）+ Quick Start + License
- **替换占位符**：所有 `<org>` → `guts-yang`（README 4 处 + pyproject.toml 2 处）
- **添加封面图**：在 README 开头插入 `docs/images/head.png`，图片下方叠加标题 + 副标题
- **砍掉内容**：长篇 Architecture mermaid、详细 Configuration 表、Evaluation 章节、Project Structure ASCII tree

## 视觉/功能效果

- 远程仓库分支列表干净，仅剩 main
- 项目结构符合 Python 社区最佳实践（src-layout）
- 缺失的配置文件补齐，新克隆者可直接运行
- README 简洁有力，封面图突出品牌，一眼即可了解项目核心价值

## 技术方案

### 1. 远程分支清理

使用 `git push origin --delete <branch>` 逐个删除远程分支。PowerShell 环境下可循环执行：

```
$branches = @(
  "dependabot/docker/docker/nginx-1.31-alpine",
  "dependabot/docker/docker/node-26-alpine",
  "dependabot/docker/docker/python-3.14-slim",
  "dependabot/github_actions/codecov/codecov-action-7",
  "dependabot/docker/docker/build-push-action-7",
  "dependabot/github_actions/docker/login-action-4",
  "dependabot/github_actions/docker/setup-buildx-action-4",
  "dependabot/github_actions/docker/setup-qemu-action-4"
)
$branches | ForEach-Object { git push origin --delete $_ }
```

### 2. 项目结构重构

#### 重命名 `gridtrace/` → `src/gridtrace/`

- 使用 `git mv gridtrace/ src/gridtrace/` 保留文件历史
- **更新 `pyproject.toml` L63**：`packages = ["src/gridtrace", "api"]`
- **更新所有 import 语句**（需扫描修改）：
- `api/` 目录下所有 `from gridtrace.xxx import ...`
- `tests/` 目录下所有 `from gridtrace.xxx import ...`
- `scripts/` 目录下所有 `from gridtrace.xxx import ...`
- `src/gridtrace/` 内部子模块的相对 import

#### 添加配置文件

- **`.github/workflows/ci.yml`**：基础 CI 工作流（lint + test），匹配 README 徽章
- **`.gitignore`**：Python 标准忽略（`__pycache__/`、`.venv/`、`*.pyc`、`.env`、`.coverage` 等）
- **`.env.example`**：从 README Configuration 表提取环境变量模板

#### 清理冗余

- 删除 `docs/api_reference.md`（README 不再引用）

### 3. README 重写策略

#### 封面图区域

```html
<p align="center">
  <img src="docs/images/head.png" alt="GridTrace" width="800">
</p>
<p align="center">
  <strong>GridTrace</strong> — Trace the path through the grid
</p>
<p align="center">
  <em>Retrieval at the fundamental particle scale</em>
</p>
```

#### 目标行数预算

- 封面图区域：8 行
- 徽章行：3 行
- Overview + 核心特性表：15 行
- Algorithm（精简为 1+2+3）：25 行
- Quick Start：30 行
- License：3 行
- **总计：约 90 行**（符合 ≤100 行要求）

#### 砍掉内容清单

- ❌ Architecture mermaid 图（37 行）
- ❌ Joint Semantic Representation 章节（11 行）
- ❌ Configuration 表（11 行）→ 移到 `.env.example`
- ❌ Evaluation 章节（11 行）→ 合并到 Quick Start 末尾
- ❌ Project Structure ASCII tree（12 行）
- ❌ Contributing 章节（4 行）→ 在 Quick Start 末尾简化为一行

## 性能与可靠性

- `git mv` 比 mv + git add 更快且保留历史
- 批量删除分支使用 ForEach-Object 管道，避免手动重复
- 重命名后需运行 `make lint test` 验证（用户确认后执行）

## 避免技术债务

- 使用 `src-layout` 是 Python 社区标准做法，避免意外从工作区导入
- `.env.example` 防止敏感配置泄露
- 完整的 `.gitignore` 防止误提交 `__pycache__`、`.env` 等文件

## 代理扩展

本次任务为本地文件操作 + Git 远程命令，无需使用任何 Skill/MCP/SubAgent 扩展。所有操作均可通过内置工具（`execute_command`、`read_file`、`write_file`）完成。