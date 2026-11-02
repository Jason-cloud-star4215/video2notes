#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2notes.py —— 视频链接 → 结构化学习讲义 PDF，全流程一键自动化

用法：
    python video2notes.py <视频链接>
    python video2notes.py <视频链接> -o "D:\\讲义" -m medium
    python video2notes.py --install-shortcut        # 在桌面创建一键快捷方式
    python video2notes.py --check                   # 只做环境自检

流程：
    解析链接 → 探测元数据 → 取字幕（无则本地语音转写）→ 清洗去重 → 抽帧
    → 调 LLM 重组成讲义 → 生成 LaTeX → 编译 PDF → 输出体检报告

依赖：yt-dlp、ffmpeg、Python 包(openai-whisper/torch/Pillow/requests)、TeX(xelatex)
      可选：DEEPSEEK_API_KEY（没有则降级为纯转写文稿 PDF）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

APP = "video2notes"
VERSION = "1.0.0"

# ---------------------------------------------------------------- 全局状态

LOG_LINES: list[str] = []
WARNINGS: list[str] = []
STATS: dict = {}


def log(msg: str = "", level: str = "info") -> None:
    """同时打印到控制台并记录到日志缓冲。"""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    LOG_LINES.append(line)
    prefix = {"info": "  ", "step": "\n▶ ", "warn": "  ! ", "ok": "  ✓ ", "err": "  ✗ "}.get(level, "  ")
    try:
        print(prefix + msg, flush=True)
    except UnicodeEncodeError:  # 某些控制台编码不支持中文
        print((prefix + msg).encode("utf-8", "replace").decode("ascii", "replace"), flush=True)
    if level == "warn":
        WARNINGS.append(msg)


def hr(title: str = "") -> None:
    log("")
    log("=" * 62)
    if title:
        log(title)
        log("=" * 62)


class Fatal(Exception):
    """无法继续的错误。"""


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 3600,
        check: bool = True, quiet: bool = False, tail: int = 0) -> subprocess.CompletedProcess:
    """运行外部命令。输出默认不进入日志，避免污染上下文。"""
    if not quiet:
        log("  $ " + " ".join(str(c) for c in cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    try:
        p = subprocess.run(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
            capture_output=True, timeout=timeout,
            encoding="utf-8", errors="ignore",
        )
    except FileNotFoundError as e:
        raise Fatal(f"命令不存在：{cmd[0]}（请先安装）") from e
    except subprocess.TimeoutExpired as e:
        raise Fatal(f"命令超时（>{timeout}s）：{cmd[0]}") from e

    if not quiet and tail and (p.stdout or p.stderr):
        text = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()
        for ln in text[-tail:]:
            log("    | " + ln[:200])
    if check and p.returncode != 0:
        detail = ((p.stderr or "") + (p.stdout or "")).strip().splitlines()
        raise Fatal(f"命令失败（返回码 {p.returncode}）：{cmd[0]}\n     " +
                    ("\n     ".join(detail[-3:]) if detail else "(无输出)"))
    return p


# ================================================================ 1. 环境自检

def which(name: str) -> str | None:
    return shutil.which(name)


def _try_import(mod: str) -> bool:
    """真实导入一次。用于识别「目录残留但文件被删」的损坏包。"""
    import importlib
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def find_xi() -> str | None:
    """找 xelatex（可能在 MiKTeX 的非 PATH 目录里）。"""
    p = which("xelatex")
    if p:
        return p
    for base in [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
                 Path("C:/texlive") , Path("C:/Program Files/MiKTeX/miktex/bin/x64")]:
        cand = base / "xelatex.exe"
        if cand.exists():
            return str(cand)
    return None


def check_env(auto_install: bool = True) -> dict:
    """检查依赖，返回环境字典。缺失的必需项会尝试自动补。"""
    hr("第 0 步 · 环境自检")
    env = {"python": sys.executable}

    # --- Python 包 ---
    import importlib.util as iu
    pkgs = {}
    for mod in ["faster_whisper", "whisper", "torch", "PIL", "requests", "numpy"]:
        pkgs[mod] = iu.find_spec(mod) is not None
    log(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    # 可选包：有 faster-whisper 时，openai-whisper / torch / Pillow 都不是必需的
    OPTIONAL = {"whisper", "torch", "PIL"}
    for m, ok in pkgs.items():
        if ok:
            log(f"  包 {m:16s} OK")
        elif m in OPTIONAL and pkgs.get("faster_whisper"):
            log(f"  包 {m:16s} 未安装（不影响：转写用 faster-whisper）")
        else:
            log(f"  包 {m:16s} 缺失", "warn")

    if not pkgs["requests"]:
        raise Fatal("缺少 requests，请运行：python -m pip install requests")
    env["has_faster_whisper"] = pkgs["faster_whisper"]
    env["has_whisper"] = pkgs["whisper"]

    # --- 关键包真实导入测试 ---
    # 有些环境下 site-packages 被清理过：目录还在、dist-info 还在，但 .py 文件已被删光，
    # 此时 find_spec 会误判为「已安装」。因此对关键包做一次真实导入。
    CRITICAL = ["requests", "idna.core", "tqdm", "filelock", "sniffio",
                "huggingface_hub", "faster_whisper"]
    broken = [m for m in CRITICAL if not _try_import(m)]
    if broken:
        log("")
        log(f"检测到 {len(broken)} 个关键包无法导入：{', '.join(broken)}", "warn")
        log("这是 site-packages 曾被清理过的典型症状（目录残留但文件缺失）。", "warn")
        pip_names = sorted({"requests", "idna", "tqdm", "filelock", "sniffio",
                            "huggingface-hub", "faster-whisper"}
                           | {m.split(".")[0].replace("_", "-") for m in broken})
        # venv 内不能使用 --user（会装到用户目录而非虚拟环境）
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        scope = [] if in_venv else ["--user"]
        log(f"正在自动修复（需要联网，可能几分钟）：{' '.join(pip_names)}", "warn")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + scope +
                           ["--force-reinstall", "--no-warn-script-location"] + pip_names,
                           capture_output=True, encoding="utf-8", errors="ignore", timeout=1200)
        except Exception as e:  # noqa: BLE001
            log(f"自动修复失败：{e}", "warn")
        still = [m for m in CRITICAL if not _try_import(m)]
        if still:
            log(f"修复后仍不可用：{', '.join(still)}", "warn")
            log("请手动运行：python -m pip install " + " ".join(scope) + " --force-reinstall "
                + " ".join(pip_names), "warn")
        else:
            log("依赖已自动修复完成", "ok")
        log("")

    if not pkgs["faster_whisper"] and not pkgs["whisper"]:
        log("转写后端均不可用：无字幕的视频将无法处理", "warn")
        log("  建议安装（含国内镜像加速）：python -m pip install faster-whisper", "warn")

    # --- 外部命令 ---
    env["yt_dlp"] = which("yt-dlp")
    if not env["yt_dlp"] and auto_install:
        log("未找到 yt-dlp，尝试自动安装 ...", "warn")
        p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--user", "yt-dlp"],
                           capture_output=True, encoding="utf-8", errors="ignore")
        env["yt_dlp"] = which("yt-dlp")
        if not env["yt_dlp"]:
            log("  pip 安装未生效，改用 `python -m yt_dlp` 方式调用")
    log(f"yt-dlp      {'OK  ' + env['yt_dlp'] if env['yt_dlp'] else '缺失（将用 python -m yt_dlp 兜底）'}")

    env["ffmpeg"] = which("ffmpeg")
    env["ffprobe"] = which("ffprobe")
    env["xelatex"] = find_xi()

    for name, key, hint in [
        ("ffmpeg", "ffmpeg", "winget install Gyan.FFmpeg  或  https://www.gyan.dev/ffmpeg/builds/"),
        ("ffprobe", "ffprobe", "随 ffmpeg 一起安装"),
        ("xelatex", "xelatex", "https://miktex.org/download （安装时选 D 盘可省 C 盘空间）"),
    ]:
        ok = env.get(key)
        log(f"{name:11s} {'OK  ' + ok if ok else '缺失'}",
            "info" if ok else "warn")
        if not ok:
            log(f"            → 安装方式：{hint}", "warn")

    env["has_ffmpeg"] = bool(env["ffmpeg"])
    env["has_tex"] = bool(env["xelatex"])

    # --- LLM key ---
    env["api_key"] = os.environ.get("DEEPSEEK_API_KEY") or _key_from_codex_config()
    env["api_base"] = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    env["api_model"] = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    if env["api_key"]:
        log(f"LLM 密钥   OK（{env['api_model']}，来自 "
            f"{'环境变量' if os.environ.get('DEEPSEEK_API_KEY') else 'Codex 配置'}）")
    else:
        log("LLM 密钥   缺失 → 将只输出「转写文稿」PDF，不做讲义重组", "warn")

    return env


def _key_from_codex_config() -> str | None:
    """从 ~/.codex/config.toml 里取 DeepSeek key（用户已配置，便于一键运行）。"""
    cfg = Path.home() / ".codex" / "config.toml"
    if not cfg.is_file():
        return None
    try:
        txt = cfg.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(r'experimental_bearer_token\s*=\s*"([^"]+)"', txt)
    if m:
        return m.group(1)
    m = re.search(r'api_key\s*=\s*"([^"]+)"', txt)
    return m.group(1) if m else None


# ================================================================ 2. 解析链接

PLATFORM_HINTS = [
    ("bilibili.com", "B站"), ("b23.tv", "B站短链"),
    ("youtube.com", "YouTube"), ("youtu.be", "YouTube"),
    ("douyin.com", "抖音"), ("xiaohongshu.com", "小红书"),
    ("zhihu.com", "知乎"), ("v.qq.com", "腾讯视频"),
]


def parse_url(url: str) -> dict:
    hr("第 1 步 · 解析链接")
    url = url.strip().strip('"').strip("'")
    if not url:
        raise Fatal("没有收到链接。用法：python video2notes.py <视频链接>")
    if not re.match(r"^https?://", url, re.I):
        raise Fatal(f"看起来不是有效链接：{url}\n（应以 http:// 或 https:// 开头）")
    platform = next((n for d, n in PLATFORM_HINTS if d in url.lower()), "通用平台")
    log(f"链接：{url}")
    log(f"识别平台：{platform}（由 yt-dlp 实际解析）")
    return {"url": url, "platform": platform}


# ================================================================ 3. 元数据

def ytdlp_cmd(env: dict) -> list[str]:
    return [env["yt_dlp"]] if env.get("yt_dlp") else [sys.executable, "-m", "yt_dlp"]


def probe(env: dict, url: str) -> dict:
    hr("第 2 步 · 探测视频信息")
    p = run(ytdlp_cmd(env) + ["--dump-single-json", "--no-warnings", "--no-playlist",
                              "--skip-download", url], quiet=True, timeout=180)
    if p.returncode != 0 or not (p.stdout or "").strip():
        detail = (p.stderr or "").strip().splitlines()
        hint = ""
        low = (p.stderr or "").lower()
        if "private" in low or "login" in low:
            hint = "\n     该视频可能需要登录态。可尝试：yt-dlp --cookies-from-browser edge <url>"
        raise Fatal("无法获取视频信息，请检查链接是否有效" + hint +
                    ("\n     " + detail[-1] if detail else ""))
    try:
        info = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise Fatal(f"视频信息解析失败：{e}") from e

    if info.get("_type") == "playlist" or info.get("entries"):
        entries = [e for e in (info.get("entries") or []) if e]
        if entries:
            log(f"检测到多 P / 播放列表（{len(entries)} 个），取第一个。"
                f"如需其他分 P，请在链接后加 ?p=N", "warn")
            info = entries[0]

    subs = sorted((info.get("subtitles") or {}).keys())
    auto = sorted((info.get("automatic_captions") or {}).keys())
    meta = {
        "id": info.get("id"),
        "title": (info.get("title") or "未命名视频").strip(),
        "uploader": (info.get("uploader") or info.get("channel") or "未知").strip(),
        "duration": int(info.get("duration") or 0),
        "description": (info.get("description") or "").strip(),
        "upload_date": info.get("upload_date") or "",
        "thumbnail": info.get("thumbnail") or "",
        "webpage_url": info.get("webpage_url") or url,
        "subtitles": subs,
        "auto_captions": auto,
        "chapters": info.get("chapters") or [],
        "view_count": info.get("view_count"),
    }
    mm, ss = divmod(meta["duration"], 60)
    log(f"标题：{meta['title']}")
    log(f"UP 主：{meta['uploader']}    时长：{mm} 分 {ss} 秒")
    log(f"字幕轨：人工 {subs if subs else '无'} / 自动 {auto[:4] if auto else '无'}")
    return meta


# ================================================================ 4. 字幕 / 转写

def try_subtitles(env: dict, meta: dict, work: Path) -> str | None:
    """优先下载平台字幕。成功返回纯文本，失败/无字幕返回 None。"""
    langs = [l for l in (meta["subtitles"] + meta["auto_captions"])
             if re.match(r"^(zh|ai-zh|zh-Hans|zh-CN|en)", l, re.I)]
    if not langs:
        log("该视频没有可用字幕轨", "warn")
        return None
    # 人工字幕优先
    prefer = [l for l in langs if l in meta["subtitles"]] or langs
    order = ",".join(dict.fromkeys(prefer + langs))
    log(f"尝试下载字幕：{order}")

    cmd = ytdlp_cmd(env) + [
        "--skip-download", "--write-subs", "--write-auto-subs",
        "--sub-langs", order, "--convert-subs", "srt",
        "--no-warnings", "-o", str(work / "src.%(ext)s"), meta["webpage_url"],
    ]
    p = run(cmd, quiet=True, timeout=300, check=False)
    srt_files = sorted(work.glob("src*.srt"))
    if not srt_files:
        log("字幕下载未产出文件（可能需要登录态），将回退到本地语音转写", "warn")
        return None
    srt = srt_files[0]
    text = srt_to_text(srt.read_text(encoding="utf-8", errors="ignore"))
    if len(text) < 50:
        log("字幕内容过短，视为不可用", "warn")
        return None
    log(f"字幕获取成功：{srt.name}（{len(text)} 字）", "ok")
    STATS["subtitle_source"] = "平台字幕"
    (work / "audio.srt").write_text(srt.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return text


def download_media(env: dict, meta: dict, work: Path, need_video: bool) -> tuple[Path | None, Path | None]:
    """下载音频（转写用）与视频（抽帧用）。"""
    if not env["has_ffmpeg"]:
        raise Fatal("需要 ffmpeg 才能提取音频/抽帧。请先安装 ffmpeg 后重试。")

    audio = work / "audio.wav"
    video = work / "video.mp4"

    if not audio.exists():
        log("下载音频 ...")
        run(ytdlp_cmd(env) + ["-f", "bestaudio/best", "-x", "--audio-format", "wav",
                              "--postprocessor-args", "-ar 16000 -ac 1",
                              "--no-warnings", "-o", str(work / "audio.%(ext)s"),
                              meta["webpage_url"]], quiet=True, timeout=1800, tail=2)
        cand = list(work.glob("audio*.wav"))
        if not cand:
            raise Fatal("音频下载失败（网络或链接问题）")
        audio = cand[0]

    if need_video and not video.exists():
        log("下载视频（用于抽帧）...")
        try:
            run(ytdlp_cmd(env) + ["-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                                  "--merge-output-format", "mp4", "--no-warnings",
                                  "-o", str(work / "video.%(ext)s"), meta["webpage_url"]],
                quiet=True, timeout=3600, tail=2)
        except Fatal as e:
            log(f"视频下载失败，将跳过抽帧：{e}", "warn")
            return audio, None
        cand = list(work.glob("video*.mp4")) + list(work.glob("video*.mkv")) + list(work.glob("video*.webm"))
        video = cand[0] if cand else None
    return audio, (video if video and video.exists() else None)


# faster-whisper 的模型仓库（走 HuggingFace，可用国内镜像加速）
FW_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

_TRANSCRIBE_HINT = ("以下是普通话技术课程讲解，请使用规范的技术术语。"
                    "涉及编程与数据库的专有名词请按标准写法输出。")


def transcribe_local(env: dict, audio: Path, work: Path, model_size: str) -> str:
    """本地语音转写。优先 faster-whisper（可用国内镜像），回退 openai-whisper。"""
    hr("第 3 步 · 本地语音转写（该视频无可用字幕）")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避重复加载 OpenMP 崩溃

    if env.get("has_faster_whisper"):
        STATS["asr_backend"] = f"faster-whisper/{model_size}"
        segs = _transcribe_faster(env, audio, work, model_size)
    elif env.get("has_whisper"):
        STATS["asr_backend"] = f"openai-whisper/{model_size}"
        log("未安装 faster-whisper，回退 openai-whisper；其官方模型源在国内常不可达，"
            "如失败请运行：python -m pip install faster-whisper", "warn")
        segs = _transcribe_openai(env, audio, work, model_size)
    else:
        raise Fatal("没有可用的转写后端。请运行：python -m pip install faster-whisper")

    if not segs:
        raise Fatal("转写结果为空，可能是音频静音或全为音乐")
    return _segs_to_srt(segs, work)


def _transcribe_faster(env: dict, audio: Path, work: Path, model_size: str) -> list[dict]:
    hf = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
    repo = FW_MODELS.get(model_size, FW_MODELS["medium"])
    log(f"后端：faster-whisper    模型：{repo}")
    log(f"镜像：{hf}   （首次运行需下载模型，之后复用本地缓存）")

    script = work / "_fw_asr.py"
    script.write_text(
        "import os, sys, json, glob\n"
        "os.environ.setdefault('HF_ENDPOINT', sys.argv[4])\n"
        "# 让 ctranslate2 找到 nvidia-* 包内的 CUDA 运行库（cublas / cudnn）\n"
        "for _p in glob.glob(os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', '*', 'bin')):\n"
        "    try:\n"
        "        os.add_dll_directory(_p)\n"
        "    except Exception:\n"
        "        pass\n"
        "    os.environ['PATH'] = _p + os.pathsep + os.environ.get('PATH', '')\n"
        "from faster_whisper import WhisperModel\n"
        "device = 'cpu'\n"
        "try:\n"
        "    import ctranslate2\n"
        "    if ctranslate2.get_cuda_device_count() > 0:\n"
        "        device = 'cuda'\n"
        "except Exception:\n"
        "    pass\n"
        "ct = 'float16' if device == 'cuda' else 'int8'\n"
        "print('device =', device, 'compute_type =', ct, flush=True)\n"
        "def run(dev, ctype):\n"
        "    mm = WhisperModel(sys.argv[3], device=dev, compute_type=ctype)\n"
        "    ss, ii = mm.transcribe(sys.argv[1], language='zh', beam_size=5,\n"
        "                           vad_filter=True, initial_prompt=sys.argv[5])\n"
        "    return [{'start': s.start, 'end': s.end, 'text': s.text} for s in ss]\n"
        "try:\n"
        "    out = run(device, ct)\n"
        "except Exception as e:\n"
        "    msg = str(e)\n"
        "    low = msg.lower()\n"
        "    if device == 'cuda' and ('cublas' in low or 'cudnn' in low or 'cuda' in low):\n"
        "        print('GPU 库缺失（' + msg[:60] + '），自动回退 CPU', flush=True)\n"
        "        print('如需 GPU 加速：pip install nvidia-cublas-cu12 nvidia-cudnn-cu12', flush=True)\n"
        "        out = run('cpu', 'int8')\n"
        "    else:\n"
        "        raise\n"
        "json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)\n"
        "print('segments =', len(out), flush=True)\n",
        encoding="utf-8")

    t0 = time.time()
    run([sys.executable, str(script), str(audio), str(work / "asr.json"), repo, hf,
         _TRANSCRIBE_HINT], timeout=10800, tail=4)
    log(f"转写完成，用时 {time.time()-t0:.0f} 秒")
    return json.loads((work / "asr.json").read_text(encoding="utf-8"))


def _transcribe_openai(env: dict, audio: Path, work: Path, model_size: str) -> list[dict]:
    log(f"后端：openai-whisper    模型：{model_size}")
    script = work / "_transcribe.py"
    script.write_text(
        "import sys, json\n"
        "import whisper\n"
        "m = whisper.load_model(sys.argv[3])\n"
        "r = m.transcribe(sys.argv[1], language='zh', verbose=False, initial_prompt=sys.argv[4])\n"
        "print('segments =', len(r.get('segments', [])), flush=True)\n"
        "json.dump(r, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)\n",
        encoding="utf-8")
    t0 = time.time()
    run([sys.executable, str(script), str(audio), str(work / "asr.json"),
         model_size, _TRANSCRIBE_HINT], timeout=10800, tail=4)
    log(f"转写完成，用时 {time.time()-t0:.0f} 秒")
    data = json.loads((work / "asr.json").read_text(encoding="utf-8"))
    return data.get("segments") or []


def _segs_to_srt(segs: list[dict], work: Path) -> str:
    srt_lines, text_parts = [], []
    n = 0
    for s in segs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        n += 1
        srt_lines.append(f"{n}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{t}\n")
        text_parts.append(f"[{fmt_ts(s['start'])}] {t}")
    (work / "audio.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    log(f"得到 {n} 段，约 {sum(len(x) for x in text_parts)} 字", "ok")
    STATS["subtitle_source"] = f"本地转写({STATS.get('asr_backend', 'faster-whisper')})"
    STATS["asr_segments"] = n
    if STATS.get("duration") and segs:
        STATS["asr_coverage"] = round(min(1.0, segs[-1]["end"] / STATS["duration"]), 3)
    return "\n".join(text_parts)


def fmt_ts(sec: float) -> str:
    mm, ss = divmod(int(sec), 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"


def srt_to_text(srt: str) -> str:
    """SRT → 带时间戳的文本，同时去重（自动字幕常有滚动重复）。"""
    out, seen = [], set()
    for block in re.split(r"\n\s*\n", srt.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        ts = lines[1] if "-->" in lines[1] else (lines[0] if "-->" in lines[0] else "")
        body = " ".join(lines[2:] if "-->" in lines[1] else lines[1:]).strip()
        body = clean_line(body)
        if not body or body in seen:
            continue
        seen.add(body)
        start = ts.split("-->")[0].strip().replace(",", ".")
        out.append(f"[{to_mmss(start)}] {body}")
    return "\n".join(out)


def to_mmss(ts: str) -> str:
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return f"{int(parts[0])*60+int(parts[1]):02d}:{int(float(parts[2])):02d}"
        if len(parts) == 2:
            return f"{int(parts[0]):02d}:{int(float(parts[1])):02d}"
    except ValueError:
        pass
    return ts[:8]


def clean_line(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)                       # HTML 标签
    s = re.sub(r"\{\\[^}]*\}", "", s)                   # {\an8} 之类
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ================================================================ 5. 抽帧

def extract_frames(env: dict, video: Path | None, meta: dict, work: Path, count: int) -> list[dict]:
    hr("第 4 步 · 抽取关键画面")
    if not video:
        log("没有视频文件，跳过抽帧", "warn")
        return []
    fdir = work / "frames"
    fdir.mkdir(exist_ok=True)
    for old in fdir.glob("*.jpg"):
        old.unlink()

    dur = meta["duration"] or 0
    if dur <= 0:
        log("时长未知，跳过抽帧", "warn")
        return []

    # 跳过片头片尾各 5%，在这个区间内均匀取帧
    lo, hi = dur * 0.05, dur * 0.95
    step = max(1.0, (hi - lo) / max(1, count))
    frames = []
    ts = lo
    while ts < hi and len(frames) < count:
        dst = fdir / f"frame_{len(frames)+1:03d}.jpg"
        p = run([env["ffmpeg"], "-ss", f"{ts:.2f}", "-i", str(video), "-frames:v", "1",
                 "-q:v", "4", "-vf", "scale=1280:-2", "-y", str(dst)],
                quiet=True, check=False, timeout=120)
        if p.returncode == 0 and dst.exists() and dst.stat().st_size > 2000:
            frames.append({"path": dst, "time": ts, "stamp": fmt_ts(ts)})
        ts += step
    log(f"抽出 {len(frames)} 帧（每 {step:.0f} 秒一帧）", "ok")
    STATS["frames"] = len(frames)
    return frames


# ================================================================ 6. LLM 重组讲义

LLM_PROMPT = """你是一位资深技术课程讲义编辑。请把下面这份视频课程的语音转写，重组成一份**结构清晰、可直接用于学习的中文讲义**。

## 视频信息
- 标题：{title}
- 作者：{uploader}
- 时长：{duration}
- 平台链接：{url}
{chapter_block}
## 语音转写（方括号内为该句在视频中的时间点）
{transcript}

## 硬性要求

1. **完整覆盖**：不遗漏任何知识点、定义、推导、例子、限制条件、口头补充的说明。宁可啰嗦，不可遗漏。你的目标是让读者不回头看视频也能学会本节全部内容。
2. **重排而非转述**：按"读者要理解什么"组织章节顺序，不要照搬转写的先后顺序。
3. **保留时间锚点**：在讲解每个知识点时，用 `（12:35）` 这样的格式标出它在视频中的位置，方便回查。
4. **数学与公式**：用 LaTeX 语法，行内写 `$x$`，独立公式写 `$$...$$`。独立公式后面必须紧跟一行符号说明，例如：`其中 $n$ 为样本数，$\mu$ 为均值。`
5. **代码**：原样保留，放入三反引号代码块，并标注语言。
6. **易错点**：如果讲者提到常见错误、隐藏前提、容易混淆的概念，单独用一段以 `> 注意：` 开头写出来。
7. **章节收尾**：每一节末尾写一段 `### 本节小结`。
8. **文末必须包含两节**：
   - `## 自测与检验`：4~6 道问答题（不用选择题），每题给出答案要点，并标注回查时间点。
   - `## 术语索引`：本讲出现的关键术语表格，列为「术语 | 所在小节 | 时间点」。

## 输出格式（严格遵守）

- 只输出 Markdown，第一行必须是 `# ` 开头的一级标题（讲义标题，含课程名与本讲主题）。
- 允许使用的语法**仅限**：`#`/`##`/`###` 三级标题、普通段落、`-` 无序列表、`1.` 有序列表、`**加粗**`、`` `行内代码` ``、三反引号代码块、`$公式$`、`$$公式$$`、Markdown 表格、`> 引用`。
- **不要**使用四级及以上标题，不要使用图片语法，不要输出任何与讲义无关的客套话或说明。
- 直接输出正文，不要写"以下是讲义"之类的开场白。
"""


def build_transcript_text(meta: dict, raw: str, frames: list[dict]) -> str:
    lines = [raw]
    if frames:
        lines.append("")
        lines.append("## 画面时间点（讲者展示的幻灯片/板书出现在这些时刻）")
        lines.append("、".join(f["stamp"] for f in frames))
    return "\n".join(lines)


def call_llm(env: dict, meta: dict, transcript: str, work: Path) -> str | None:
    hr("第 5 步 · 重组讲义（调用语言模型）")
    if not env.get("api_key"):
        log("未配置 LLM 密钥，跳过讲义重组，改为输出转写文稿", "warn")
        return None

    ch = meta.get("chapters") or []
    chapter_block = ""
    if ch:
        chapter_block = "- 官方章节：\n" + "\n".join(
            f"    - {c.get('title','')}（{fmt_ts(c.get('start_time',0))}）" for c in ch[:30]) + "\n"

    prompt = LLM_PROMPT.format(
        title=meta["title"], uploader=meta["uploader"],
        duration=f"{meta['duration']//60} 分 {meta['duration']%60} 秒",
        url=meta["webpage_url"], chapter_block=chapter_block,
        transcript=transcript,
    )
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    STATS["prompt_chars"] = len(prompt)

    body = {
        "model": env["api_model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
    }
    url = env["api_base"].rstrip("/") + "/chat/completions"

    last_err = ""
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {env['api_key']}"})
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read().decode("utf-8"))
            md = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage") or {}
            if usage:
                STATS["llm_usage"] = usage
                log(f"token 用量：输入 {usage.get('prompt_tokens','?')} / "
                    f"输出 {usage.get('completion_tokens','?')}"
                    + (f"（缓存命中 {usage.get('prompt_cache_hit_tokens')}）"
                       if usage.get("prompt_cache_hit_tokens") is not None else ""))
            if not md.strip():
                last_err = "模型返回空内容"
                continue
            (work / "notes_raw.md").write_text(md, encoding="utf-8")
            log(f"讲义生成完成：{len(md)} 字符，约 {len(re.findall(r'[一-龥]', md))} 汉字", "ok")
            STATS["llm_chars"] = len(md)
            STATS["llm_han"] = len(re.findall(r"[一-龥]", md))
            return md
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            log(f"第 {attempt} 次调用失败：{last_err}", "warn")
            if attempt < 3:
                time.sleep(3 * attempt)
    log(f"LLM 调用最终失败（{last_err}），将降级为输出转写文稿", "warn")
    return None


# ================================================================ 7. Markdown → LaTeX

LATEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}"}


def esc(t: str) -> str:
    return "".join(LATEX_ESC.get(c, c) for c in t)


def inline(t: str) -> str:
    """行内转换：先抽出公式与代码，转义其余文本，再套 Markdown 语法。"""
    stash: list[str] = []

    def keep(m, template):
        stash.append(template.format(m.group(1)))
        return f"\x00{len(stash)-1}\x00"

    t = re.sub(r"\$([^$]+)\$", lambda m: keep(m, "${0}$"), t)
    t = re.sub(r"`([^`]+)`", lambda m: keep(m, r"\texttt{{{0}}}"), t)
    t = esc(t)
    # esc() 不处理 *，因此加粗标记在转义后仍然完好，可以直接替换
    t = re.sub(r"\*\*([^*]+)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", lambda m: r"\textit{" + m.group(1) + "}", t)
    for i, s in enumerate(stash):
        t = t.replace(f"\x00{i}\x00", s)
    return t


def md_table_to_latex(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    if len(cells) >= 2 and all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells[1]):
        header, body = cells[0], cells[2:]
    else:
        header, body = cells[0], cells[1:]
    n = len(header)
    spec = "p{%.2f\\textwidth}" % (0.94 / max(n, 1))
    out = [r"\begin{center}", r"\begin{tabular}{" + spec * n + "}", r"\toprule",
           " & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\", r"\midrule"]
    for row in body:
        row = (row + [""] * n)[:n]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    return "\n".join(out)


def md_to_latex(md: str) -> str:
    """把受控 Markdown 转成可编译的 LaTeX 片段。"""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    in_code = in_ul = in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append(r"\end{itemize}"); in_ul = False
        if in_ol:
            out.append(r"\end{enumerate}"); in_ol = False

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        # 代码块
        if s.startswith("```"):
            if in_code:
                out.append(r"\end{lstlisting}")
                in_code = False
            else:
                close_lists()
                lang = s[3:].strip() or ""
                out.append(r"\begin{lstlisting}" + (f"[language={lang}]" if lang else "") )
                in_code = True
            i += 1
            continue
        if in_code:
            out.append(ln)
            i += 1
            continue

        if not s:
            close_lists(); out.append(""); i += 1; continue

        # 表格
        if s.startswith("|") and s.endswith("|"):
            close_lists()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            if len(block) >= 2:
                out.append(md_table_to_latex(block))
            else:
                out.append(inline(s))
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s*(.+)$", s)
        if m:
            close_lists()
            lvl, txt = len(m.group(1)), inline(m.group(2))
            if lvl == 1:
                pass  # 一级标题已在标题页体现，正文略过
            elif lvl == 2:
                out.append(f"\\section{{{txt}}}")
            elif lvl == 3:
                out.append(f"\\subsection{{{txt}}}")
            else:
                out.append(f"\\subsubsection{{{txt}}}")
            i += 1
            continue

        # 引用 / 提示框
        if s.startswith(">"):
            close_lists()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            body = " ".join(x for x in buf if x)
            box = "warningbox" if re.search(r"注意|易错|坑|错误|小心|陷阱", body) else "knowledgebox"
            title = "注意" if box == "warningbox" else "补充"
            out.append(f"\\begin{{{box}}}{{{title}}}")
            out.append(inline(body))
            out.append(f"\\end{{{box}}}")
            out.append("")
            continue

        # 无序列表
        m = re.match(r"^[-*+]\s+(.*)$", s)
        if m:
            if in_ol:
                out.append(r"\end{enumerate}"); in_ol = False
            if not in_ul:
                out.append(r"\begin{itemize}"); in_ul = True
            out.append(r"  \item " + inline(m.group(1)))
            i += 1
            continue

        # 有序列表
        m = re.match(r"^\d+[.)]\s+(.*)$", s)
        if m:
            if in_ul:
                out.append(r"\end{itemize}"); in_ul = False
            if not in_ol:
                out.append(r"\begin{enumerate}"); in_ol = True
            out.append(r"  \item " + inline(m.group(1)))
            i += 1
            continue

        # 独立公式
        if s.startswith("$$"):
            close_lists()
            buf = [s]
            if not (s.endswith("$$") and len(s) > 4):
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    buf.append(lines[i]); i += 1
                if i < len(lines):
                    buf.append(lines[i])
            out.append("\\[" + "\n".join(buf).strip().strip("$") + "\\]")
            i += 1
            continue

        close_lists()
        out.append(inline(s))
        out.append("")
        i += 1

    if in_code:
        out.append(r"\end{lstlisting}")
    close_lists()
    return "\n".join(out)


# ================================================================ 8. LaTeX 文档

PREAMBLE = r"""\documentclass[a4paper]{article}

% —— 中文：优先使用 Windows 系统中文字体，保证 PDF 可检索、可复制 ——
\IfFileExists{ctex.sty}{
  \usepackage{fontspec}
  \IfFontExistsTF{SimSun}{
    \usepackage[fontset=windows]{ctex}
  }{
    \usepackage[fontset=fandol]{ctex}
  }
}{
  \usepackage{fontspec}
  \XeTeXlinebreaklocale "zh"
  \XeTeXlinebreakskip = 0pt plus 1pt
  \IfFontExistsTF{Noto Serif CJK SC}{\setmainfont{Noto Serif CJK SC}}{}
}

\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage[margin=2.3cm]{geometry}
\usepackage[most]{tcolorbox}
\usepackage{etoolbox}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{float}
\usepackage{enumitem}
\usepackage{fancyhdr}

\hypersetup{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}

\newtcolorbox{knowledgebox}[1]{
    enhanced, colback=blue!5!white, colframe=blue!75!black, colbacktitle=blue!75!black,
    coltitle=white, fonttitle=\bfseries, title=#1, sharp corners, breakable,
    boxrule=0.8pt, left=6pt, right=6pt, top=5pt, bottom=5pt
}
\newtcolorbox{importantbox}[1]{
    enhanced, colback=yellow!10!white, colframe=yellow!80!black, colbacktitle=yellow!80!black,
    coltitle=black, fonttitle=\bfseries, title=#1, sharp corners, breakable,
    boxrule=0.8pt, left=6pt, right=6pt, top=5pt, bottom=5pt
}
\newtcolorbox{warningbox}[1]{
    enhanced, colback=red!4!white, colframe=red!70!black, colbacktitle=red!70!black,
    coltitle=white, fonttitle=\bfseries, title=#1, sharp corners, breakable,
    boxrule=0.8pt, left=6pt, right=6pt, top=5pt, bottom=5pt
}

\lstset{
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue!70!black},
    stringstyle=\color{red!60!black},
    commentstyle=\color{green!45!black},
    breaklines=true, frame=single, numbers=left,
    numberstyle=\tiny\color{gray}, captionpos=b, extendedchars=false,
    showstringspaces=false
}

\newcommand{\vtag}{\protect\footnotemark}
\newcommand{\srcnote}[1]{\footnotetext{视频画面时间区间：#1}}
\newcommand{\degC}{\ensuremath{^{\circ}\mathrm{C}}}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\small\thepage}
\fancyhead[L]{\small\notetitle}
\renewcommand{\headrulewidth}{0.4pt}

\newcommand{\notetitle}{__TITLE__}

\begin{document}

\begin{titlepage}
\centering
{\Large 视频学习讲义\par}
\vspace{1.2cm}
{\huge\bfseries \notetitle\par}
\vspace{0.6cm}
{\large __UPLOADER__\par}
\vspace{0.3cm}
{\large __DATE__\par}
\vspace{0.9cm}
__COVERBLOCK__
\vfill
\begin{tcolorbox}[width=0.92\textwidth, colback=black!2!white, colframe=black!55, sharp corners]
\textbf{视频作者}：__UPLOADER__\par
\textbf{视频时长}：__DURATION__\par
\textbf{字幕来源}：__SUBSRC__\par
\textbf{原视频}：\href{__URL__}{\nolinkurl{__URL__}}\par
\textbf{生成时间}：__GENTIME__\par
\end{tcolorbox}
\end{titlepage}

\tableofcontents
\newpage

__BODY__

\end{document}
"""


def build_tex(meta: dict, body_tex: str, cover: str, work: Path, src: str) -> Path:
    cover_block = ""
    if cover:
        cover_block = (r"\includegraphics[width=0.82\textwidth,height=0.38\textheight,"
                       r"keepaspectratio]{" + cover + r"}\par")
    doc = (PREAMBLE
           .replace("__COVERBLOCK__", cover_block)
           .replace("__TITLE__", esc(meta["title"]))
           .replace("__UPLOADER__", esc(meta["uploader"]))
           .replace("__DATE__", datetime.now().strftime("%Y 年 %m 月 %d 日"))
           .replace("__GENTIME__", datetime.now().strftime("%Y-%m-%d %H:%M"))
           .replace("__DURATION__", f"{meta['duration']//60} 分 {meta['duration']%60} 秒")
           .replace("__SUBSRC__", esc(src))
           .replace("__URL__", meta["webpage_url"])
           .replace("__BODY__", body_tex))
    p = work / "notes.tex"
    p.write_text(doc, encoding="utf-8")
    return p


def compile_pdf(env: dict, tex: Path, work: Path) -> Path | None:
    hr("第 6 步 · 编译 PDF")
    if not env["has_tex"]:
        log("未找到 xelatex，跳过编译。LaTeX 源码已保留：" + str(tex), "warn")
        return None
    for i in (1, 2):
        p = run([env["xelatex"], "-interaction=nonstopmode", "-halt-on-error",
                 tex.name], cwd=work, check=False, timeout=900, quiet=True)
        if p.returncode != 0:
            err = re.findall(r"^!.*$|^l\.\d+.*$", p.stdout or "", re.M)
            log(f"第 {i} 遍编译失败", "warn")
            for e in err[:6]:
                log("    " + e[:160])
            log("LaTeX 源码已保留，可用 MiKTeX 控制台查看完整日志", "warn")
            if (work / "notes.pdf").exists():
                log("使用了上一遍生成的部分 PDF", "warn")
            return (work / "notes.pdf") if (work / "notes.pdf").exists() else None
    pdf = work / "notes.pdf"
    if pdf.exists():
        log(f"PDF 生成成功：{pdf.name}（{pdf.stat().st_size/1024:.0f} KB）", "ok")
        return pdf
    log("编译未产出 PDF", "warn")
    return None


def download_cover(meta: dict, work: Path) -> str:
    if not meta.get("thumbnail"):
        return ""
    try:
        dst = work / "cover.jpg"
        req = urllib.request.Request(meta["thumbnail"], headers={"User-Agent": "Mozilla/5.0",
                                                                 "Referer": "https://www.bilibili.com/"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dst, "wb") as f:
            f.write(r.read())
        if dst.stat().st_size > 1000:
            log("封面已下载")
            return "cover.jpg"
    except Exception as e:  # noqa: BLE001
        log(f"封面下载失败（不影响结果）：{e}", "warn")
    return ""


# ================================================================ 9. 体检报告

def write_report(work: Path, meta: dict, env: dict, pdf: Path | None,
                 frames: list[dict], elapsed: float, degraded: bool) -> Path:
    """产出一份「这份讲义靠不靠谱」的自检报告。"""
    pdf_pages = pdf_han = None
    if pdf and pdf.exists():
        pt = which("pdftotext")
        if pt:
            try:
                r = subprocess.run([pt, str(pdf), "-"], capture_output=True, timeout=180)
                out = r.stdout.decode("utf-8", errors="ignore")
                pdf_han = len(re.findall(r"[一-龥]", out))
                pdf_pages = out.count("\f") or None
            except Exception:  # noqa: BLE001
                pass

    lines = [
        f"{APP} v{VERSION} · 体检报告",
        "=" * 60,
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"总耗时：{elapsed/60:.1f} 分钟",
        "",
        "【视频】",
        f"  标题：{meta['title']}",
        f"  UP 主：{meta['uploader']}",
        f"  时长：{meta['duration']//60} 分 {meta['duration']%60} 秒",
        f"  链接：{meta['webpage_url']}",
        "",
        "【内容来源】",
        f"  字幕来源：{STATS.get('subtitle_source', '未获取')}",
    ]
    if STATS.get("asr_segments"):
        lines.append(f"  转写段数：{STATS['asr_segments']}")
    if STATS.get("asr_coverage") is not None:
        c = STATS["asr_coverage"]
        flag = "正常" if c > 0.9 else "偏低，可能有大段未转写内容"
        lines.append(f"  转写覆盖率：{c:.1%}  → {flag}")
    if STATS.get("llm_usage"):
        u = STATS["llm_usage"]
        lines.append(f"  LLM token：输入 {u.get('prompt_tokens','?')} / "
                     f"输出 {u.get('completion_tokens','?')}")
        if u.get("prompt_cache_hit_tokens") is not None:
            lines.append(f"    其中缓存命中输入：{u.get('prompt_cache_hit_tokens')}")
    if STATS.get("llm_han"):
        lines.append(f"  讲义正文：{STATS['llm_han']} 汉字")
        note_chars = sum(len(f"[{f['stamp']}]") for f in frames)
        lines.append(f"  视频转写字数：{STATS.get('transcript_han', '?')} 汉字")
        ratio = STATS["llm_han"] / max(1, STATS.get("transcript_han", 1))
        verdict = "正常（讲义在重排与补充结构性文字）" if ratio < 1.6 else "偏高，请抽查是否有模型自行补充的内容"
        lines.append(f"  讲义/转写 比：{ratio:.2f}  → {verdict}")
    lines += [
        "",
        "【配图】",
        f"  抽取帧数：{len(frames)}",
        f"  插入讲义：见正文（每张标注视频时间点）",
        "",
        "【成品】",
    ]
    if pdf:
        lines.append(f"  PDF：{pdf.name}  {pdf.stat().st_size/1024:.0f} KB")
        if pdf_pages:
            lines.append(f"  页数：约 {pdf_pages} 页")
        if pdf_han:
            ok = pdf_han > 200
            lines.append(f"  PDF 可提取汉字：{pdf_han}  → {'可检索、可复制' if ok else '⚠ 无法检索（字体问题）'}")
        else:
            lines.append("  PDF 文本提取：未检测（缺少 pdftotext）")
    else:
        lines.append("  ⚠ 未生成 PDF")
    if degraded:
        lines.append("  ⚠ 本次为降级产物（未做讲义重组）")

    lines += ["", "【需要你留意的地方】"]
    if WARNINGS:
        for w in WARNINGS[-12:]:
            lines.append(f"  - {w}")
    else:
        lines.append("  （无）")
    lines += [
        "",
        "【建议的下一步】",
        "  1. 先通读讲义，对照文末「自测与检验」自问一遍——答不出的地方回看对应时间点。",
        "  2. 抽查 2~3 张配图，确认画面与所属章节对得上。",
        "  3. 隔几天再做一遍自测题；分散重做比连续重复有效。",
        "",
        "  判断这份讲义值不值得深挖的标准，不是页数，而是：",
        "  合上讲义，能不能用自己的话把它讲一遍。",
    ]

    p = work / "体检报告.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    (work / "run.log").write_text("\n".join(LOG_LINES), encoding="utf-8")
    return p


# ================================================================ 10. 主流程

def safe_name(s: str, limit: int = 40) -> str:
    """把标题变成安全的目录名：去掉特殊字符，并限制长度（避免 Windows 路径超长）。"""
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', " ", s)
    s = re.sub(r"[#＃\[\]【】()（）]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > limit:
        s = s[:limit].rstrip(" .，,、")
    return s or "未命名视频"


def ensure_short_path(p: Path) -> Path:
    """若路径过长，退回到更短的目录名（Windows 默认上限 260 字符）。"""
    if len(str(p)) <= 180:
        return p
    short = p.parent / (p.name[:24].rstrip(" .") + "_" + str(abs(hash(p.name)) % 100000))
    log(f"路径过长，改用短目录名：{short.name}", "warn")
    return short


def process(env: dict, url: str, out_root: Path, model_size: str, frames_n: int,
            keep_media: bool) -> tuple[Path, bool]:
    t0 = time.time()
    parsed = parse_url(url)
    meta = probe(env, parsed["url"])
    STATS["duration"] = meta["duration"]

    work = ensure_short_path(out_root / safe_name(meta["title"]))
    work.mkdir(parents=True, exist_ok=True)
    log(f"输出目录：{work}")
    (work / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 字幕
    hr("第 3 步 · 获取讲解文本")
    raw = try_subtitles(env, meta, work)
    if raw is None:
        audio, _ = download_media(env, meta, work, need_video=False)
        raw = transcribe_local(env, audio, work, model_size)
    STATS["transcript_han"] = len(re.findall(r"[一-龥]", raw))
    (work / "transcript_indexed.txt").write_text(raw, encoding="utf-8")

    # 抽帧
    _, video = download_media(env, meta, work, need_video=True)
    frames = extract_frames(env, video, meta, work, frames_n)
    cover = download_cover(meta, work)

    # 讲义
    degraded = False
    if meta.get("description"):
        (work / "视频简介.txt").write_text(meta["description"], encoding="utf-8")

    md = call_llm(env, meta, build_transcript_text(meta, raw, frames), work)
    if md is None:
        degraded = True
        md = compose_fallback(meta, raw)

    body = md_to_latex(md) if not degraded else fallback_to_latex(meta, raw)
    tex = build_tex(meta, body, cover, work, STATS.get("subtitle_source", "未知"))
    pdf = compile_pdf(env, tex, work)

    # 清理过程媒体
    if not keep_media and pdf:
        freed = 0
        for f in ["video.mp4", "audio.wav", "asr.json", "_transcribe.py"]:
            fp = work / f
            if fp.exists():
                freed += fp.stat().st_size
                fp.unlink()
        fdir = work / "frames"
        if fdir.exists():
            for fp in fdir.glob("*.jpg"):
                freed += fp.stat().st_size
                fp.unlink()
        if freed:
            log(f"已清理过程文件，释放 {freed/1048576:.1f} MB（加 --keep-media 可保留）")

    rep = write_report(work, meta, env, pdf, frames, time.time() - t0, degraded)
    hr("完成")
    log(f"输出目录：{work}")
    if pdf:
        log(f"讲义 PDF：{pdf.name}", "ok")
    log(f"体检报告：{rep.name}", "ok")
    return work, degraded


def compose_fallback(meta: dict, raw: str) -> str:
    """没有 LLM 时，把转写整理成一份可读文稿。"""
    lines = [f"# {meta['title']}", "", "> 未配置语言模型密钥，本文件为原始转写文稿（已按时间分段）。",
             "> 配置 DEEPSEEK_API_KEY 后可自动重组成结构化讲义。", ""]
    lines.append("## 转写文稿")
    lines.append("")
    for ln in raw.splitlines():
        lines.append(f"- {ln}" if ln.strip() else "")
    return "\n".join(lines)


def fallback_to_latex(meta: dict, raw: str) -> str:
    out = [r"\section{转写文稿}",
           r"\begin{knowledgebox}{说明}",
           "未配置语言模型密钥，本讲义为原始语音转写文稿（按时间点分段）。",
           "在环境变量中设置 \\texttt{DEEPSEEK\\_API\\_KEY} 后可自动重组成结构化讲义。",
           r"\end{knowledgebox}", ""]
    for ln in raw.splitlines():
        m = re.match(r"\[([\d:]+)\]\s*(.*)", ln)
        if m:
            out.append(f"\\textbf{{{m.group(1)}}}　{inline(m.group(2))}\\\\")
    return "\n".join(out)


# ================================================================ 桌面快捷方式

def install_shortcut(script: Path) -> Path:
    """在桌面创建一键运行的 .bat（可拖拽链接，也可双击后粘贴）。"""
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    desktop.mkdir(exist_ok=True)
    bat = desktop / "视频转讲义.bat"
    py = sys.executable
    content = f"""@echo off
chcp 65001 >nul
title 视频转讲义
setlocal

set "SCRIPT={script}"
set "PYTHON={py}"
set "URL=%~1"

if "%URL%"=="" (
  echo ============================================
  echo   视频转讲义 - 一键生成学习 PDF
  echo ============================================
  echo.
  echo 提示：也可以直接把视频链接拖到本文件上运行。
  echo.
  set /p "URL=请粘贴视频链接后按回车："
)

if "%URL%"=="" (
  echo 未输入链接，退出。
  pause
  exit /b 1
)

echo.
echo 开始处理：%URL%
echo.
"%PYTHON%" "%SCRIPT%" "%URL%"

echo.
echo ============================================
echo 处理结束。输出目录见上方提示。
echo ============================================
pause
"""
    bat.write_text(content, encoding="utf-8")
    return bat


# ================================================================ CLI

def main() -> int:
    ap = argparse.ArgumentParser(
        prog=APP,
        description="视频链接 → 结构化学习讲义 PDF（一键全流程）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python video2notes.py "https://www.bilibili.com/video/BVxxxxxxxx"
  python video2notes.py "https://youtu.be/xxxx" -o "D:\\讲义" -m small
  python video2notes.py --install-shortcut
  python video2notes.py --check
""")
    ap.add_argument("url", nargs="?", help="视频链接（可省略，配合 --install-shortcut / --check 使用）")
    ap.add_argument("-o", "--output", default=str(Path.home() / "Desktop" / "视频讲义"),
                    help="输出根目录（默认：桌面\\视频讲义）")
    ap.add_argument("-m", "--model", default="large-v3-turbo",
                    choices=["tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"],
                    help="本地转写模型（默认 large-v3-turbo，速度与质量兼顾；有字幕时不使用）")
    ap.add_argument("-f", "--frames", type=int, default=12, help="抽帧数量（默认 12）")
    ap.add_argument("--keep-media", action="store_true", help="保留视频/音频/帧等过程文件")
    ap.add_argument("--install-shortcut", action="store_true", help="在桌面创建一键快捷方式后退出")
    ap.add_argument("--check", action="store_true", help="只做环境自检后退出")
    args = ap.parse_args()

    print(f"\n{APP} v{VERSION} — 视频转学习讲义\n")

    if args.install_shortcut:
        bat = install_shortcut(Path(__file__).resolve())
        print(f"已创建桌面快捷方式：{bat}")
        print("用法：双击后在提示下粘贴链接，或把链接直接拖到这个文件上。\n")
        return 0

    env = check_env()
    if args.check:
        print("\n环境自检结束。\n")
        return 0
    if not args.url:
        ap.print_help()
        return 1

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        work, degraded = process(env, args.url, out_root, args.model, args.frames, args.keep_media)
    except Fatal as e:
        hr("中止")
        log(str(e), "err")
        log("")
        log("排查建议：")
        log("  · 链接无效 / 视频被删除 / 需要登录 → 换一个链接或先用浏览器确认可播放")
        log("  · 网络失败 → 检查网络与代理，稍后重试")
        log("  · 缺少 ffmpeg 或 xelatex → 按上面提示安装后重试")
        return 2
    except KeyboardInterrupt:
        log("\n用户中断", "warn")
        return 130
    except Exception as e:  # noqa: BLE001
        hr("意外错误")
        log(f"{type(e).__name__}: {e}", "err")
        log("以上为程序内部错误。日志已尽量保留在输出目录中。")
        return 3

    if degraded:
        log("")
        log("提示：本次未做讲义重组（缺 API 密钥或调用失败），产出为转写文稿 PDF。", "warn")
        log("      设置环境变量 DEEPSEEK_API_KEY 后重跑即可得到完整讲义。", "warn")
    log(f"\n全部完成。目录：{work}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
