#!/usr/bin/env python3
"""Build a single self-contained HTML page for browsing per-checkpoint audio samples.

Layout:
    <samples_dir>/<label>/<prompt>.wav
where label sorts by step (e.g. step_0001000, best_step_0003900_loss_0.831009).

Produces an HTML file with a slider that switches between checkpoints and a
waveform + play button per prompt (powered by wavesurfer.js from a CDN).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

LABEL_RE = re.compile(r"^(?:best_)?step_(\d+)(?:_loss_(\d+\.\d+))?$")


def collect(samples_dir: Path) -> tuple[list[dict], list[str]]:
    entries: list[tuple[int, str, float | None, Path]] = []
    prompts: set[str] = set()
    for child in samples_dir.iterdir():
        if not child.is_dir():
            continue
        m = LABEL_RE.match(child.name)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2)) if m.group(2) else None
        entries.append((step, child.name, loss, child))
        for wav in child.glob("*.wav"):
            prompts.add(wav.stem)
    entries.sort(key=lambda t: (t[0], 0 if t[2] is None else 1))

    sorted_prompts = sorted(prompts)
    checkpoints: list[dict] = []
    for step, label, loss, dir_path in entries:
        wavs: dict[str, str] = {}
        for prompt in sorted_prompts:
            wav_path = dir_path / f"{prompt}.wav"
            if wav_path.exists():
                data = wav_path.read_bytes()
                wavs[prompt] = base64.b64encode(data).decode("ascii")
        checkpoints.append(
            {
                "label": label,
                "step": step,
                "loss": loss,
                "is_best": label.startswith("best_"),
                "wavs": wavs,
            }
        )
    return checkpoints, sorted_prompts


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://unpkg.com/wavesurfer.js@7"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; max-width: 980px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 18px; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  .header {{ display: flex; align-items: baseline; gap: 16px; margin-bottom: 12px; padding: 12px 16px; background: #fff; border: 1px solid #ddd; border-radius: 8px; }}
  .header .label {{ font-weight: 600; font-family: ui-monospace, monospace; font-size: 15px; }}
  .header .step {{ color: #666; }}
  .header .loss {{ color: #2a7; font-weight: 600; }}
  .header .best-badge {{ background: #2a7; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 11px; }}
  .slider-wrap {{ margin: 16px 0 24px; padding: 12px 16px; background: #fff; border: 1px solid #ddd; border-radius: 8px; }}
  .slider-wrap input[type=range] {{ width: 100%; }}
  .ticks {{ display: flex; justify-content: space-between; font-size: 10px; color: #888; margin-top: 4px; font-family: ui-monospace, monospace; }}
  .nav-buttons {{ display: flex; gap: 8px; margin-top: 8px; justify-content: center; }}
  .nav-buttons button {{ padding: 4px 12px; cursor: pointer; }}
  .prompt {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .prompt h2 {{ margin: 0 0 8px; font-size: 14px; color: #333; }}
  .waveform {{ height: 96px; background: #f4f4f4; border-radius: 4px; }}
  .controls {{ margin-top: 8px; display: flex; gap: 8px; align-items: center; }}
  .controls button {{ padding: 4px 12px; cursor: pointer; }}
  .duration {{ color: #888; font-size: 12px; font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{n_ckpt} checkpoints × {n_prompt} prompts. Drag the slider or use ←/→ keys to switch checkpoint.</div>

<div class="header">
  <span class="label" id="cur-label">--</span>
  <span class="step" id="cur-step">step --</span>
  <span class="loss" id="cur-loss"></span>
  <span class="best-badge" id="cur-best" style="display:none">BEST</span>
</div>

<div class="slider-wrap">
  <input type="range" id="ckpt-slider" min="0" max="{max_idx}" value="0" step="1">
  <div class="ticks" id="ticks"></div>
  <div class="nav-buttons">
    <button id="prev-btn">← prev</button>
    <button id="next-btn">next →</button>
    <button id="play-all-btn">play all prompts</button>
  </div>
</div>

<div id="prompts-container"></div>

<script>
const CHECKPOINTS = {data_json};
const PROMPTS = {prompts_json};

const container = document.getElementById('prompts-container');
const wavesurfers = {{}};
const buttons = {{}};
const durations = {{}};

PROMPTS.forEach(name => {{
  const block = document.createElement('div');
  block.className = 'prompt';
  block.innerHTML = `
    <h2>${{name}}</h2>
    <div class="waveform" id="wave-${{name}}"></div>
    <div class="controls">
      <button data-prompt="${{name}}" class="play-btn">▶ play</button>
      <span class="duration" id="dur-${{name}}">0.00s</span>
    </div>
  `;
  container.appendChild(block);

  const ws = WaveSurfer.create({{
    container: `#wave-${{name}}`,
    waveColor: '#4a90e2',
    progressColor: '#1e5fa8',
    cursorColor: '#222',
    height: 96,
    barWidth: 2,
    barGap: 1,
    barRadius: 1,
  }});
  wavesurfers[name] = ws;
  ws.on('ready', () => {{
    document.getElementById(`dur-${{name}}`).textContent = ws.getDuration().toFixed(2) + 's';
  }});
  ws.on('finish', () => {{
    const btn = document.querySelector(`.play-btn[data-prompt="${{name}}"]`);
    if (btn) btn.textContent = '▶ play';
  }});
}});

document.querySelectorAll('.play-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const name = btn.dataset.prompt;
    const ws = wavesurfers[name];
    if (ws.isPlaying()) {{
      ws.pause();
      btn.textContent = '▶ play';
    }} else {{
      ws.play();
      btn.textContent = '⏸ pause';
    }}
  }});
}});

document.getElementById('play-all-btn').addEventListener('click', () => {{
  PROMPTS.forEach(name => {{
    wavesurfers[name].stop();
    wavesurfers[name].play();
    const btn = document.querySelector(`.play-btn[data-prompt="${{name}}"]`);
    if (btn) btn.textContent = '⏸ pause';
  }});
}});

function b64ToBlob(b64) {{
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], {{ type: 'audio/wav' }});
}}

function loadCheckpoint(idx) {{
  const ckpt = CHECKPOINTS[idx];
  document.getElementById('cur-label').textContent = ckpt.label;
  document.getElementById('cur-step').textContent = `step ${{ckpt.step}}`;
  document.getElementById('cur-loss').textContent = ckpt.loss !== null ? `val_loss=${{ckpt.loss.toFixed(6)}}` : '';
  document.getElementById('cur-best').style.display = ckpt.is_best ? 'inline-block' : 'none';
  PROMPTS.forEach(name => {{
    const ws = wavesurfers[name];
    ws.stop();
    const blob = b64ToBlob(ckpt.wavs[name]);
    ws.loadBlob(blob);
    const btn = document.querySelector(`.play-btn[data-prompt="${{name}}"]`);
    if (btn) btn.textContent = '▶ play';
  }});
}}

const slider = document.getElementById('ckpt-slider');
slider.addEventListener('input', () => loadCheckpoint(parseInt(slider.value, 10)));
document.getElementById('prev-btn').addEventListener('click', () => {{
  slider.value = Math.max(0, parseInt(slider.value, 10) - 1);
  loadCheckpoint(parseInt(slider.value, 10));
}});
document.getElementById('next-btn').addEventListener('click', () => {{
  slider.value = Math.min(CHECKPOINTS.length - 1, parseInt(slider.value, 10) + 1);
  loadCheckpoint(parseInt(slider.value, 10));
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') {{ document.getElementById('prev-btn').click(); }}
  else if (e.key === 'ArrowRight') {{ document.getElementById('next-btn').click(); }}
  else if (e.key === ' ') {{ e.preventDefault(); document.querySelector('.play-btn').click(); }}
}});

const ticksEl = document.getElementById('ticks');
CHECKPOINTS.forEach(c => {{
  const span = document.createElement('span');
  span.textContent = c.is_best ? `★${{c.step}}` : c.step;
  if (c.is_best) span.style.color = '#2a7';
  ticksEl.appendChild(span);
}});

loadCheckpoint(0);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Checkpoint Sample Browser")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).resolve()
    if not samples_dir.is_dir():
        raise FileNotFoundError(samples_dir)

    checkpoints, prompts = collect(samples_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint subdirs found under {samples_dir}")

    html = HTML_TEMPLATE.format(
        title=args.title,
        n_ckpt=len(checkpoints),
        n_prompt=len(prompts),
        max_idx=len(checkpoints) - 1,
        data_json=json.dumps(checkpoints, ensure_ascii=False),
        prompts_json=json.dumps(prompts, ensure_ascii=False),
    )
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"wrote {out_path} ({size_mb:.1f} MB, {len(checkpoints)} ckpts × {len(prompts)} prompts)")


if __name__ == "__main__":
    main()
