# V4 Large 話者 LoRA 学習

この文書は、公開済み話者の一覧から学習対象を選び、対応する音声データを用いて Irodori-TTS の話者 LoRA を学習し、モデルの世代別に保存する手順を記録するものです。学習の進捗は Atmos に送ります。対象の一覧や使用するモデルを変更するときも、同じ順序で確認します。

## 対象と入力

話者 ID の基準は、[ultemica/irodori-tts](https://huggingface.co/ultemica/irodori-tts) の `v4.1-small/<category>/<speaker>.safetensors` です。公開済みファイルからその都度 ID を取得し、前回の一覧を固定値として使い回さないでください。V4 Large のベースは [Aratako/Irodori-TTS-v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) の量子化していない `model.safetensors` です。Small 用の設定ファイルやチェックポイントを混ぜてはいけません。

| 区分 | 話者 ID | 日本語音声の元データ | 2026年9月時点の公開済み対象 |
|---|---|---|---:|
| 原神 | `gi_*` | [ultemica/genshin-impact-voices](https://huggingface.co/datasets/ultemica/genshin-impact-voices) | 111 |
| 崩壊:スターレイル | `hsr_*` | [ultemica/honkai-star-rail-voices](https://huggingface.co/datasets/ultemica/honkai-star-rail-voices) | 61 |
| 鳴潮 | `wuwa_*` | [ultemica/wuthering-waves-voices](https://huggingface.co/datasets/ultemica/wuthering-waves-voices) | 23 |
| 魔法少女ノ魔女裁判 | `mgwt_*` | [ultemica/magical-girl-witch-trials-voice](https://huggingface.co/datasets/ultemica/magical-girl-witch-trials-voice) | 14 |
| VTuber | `vtuber_cherry`、`vtuber_vivi` | [ultemica/vtuber_voices](https://huggingface.co/datasets/ultemica/vtuber_voices) | 2 |

最初の4区分で209話者です。VTuberの2話者は、音声と書き起こしが対応した `manifest.jsonl` と、それが参照する `latents/*.pt` が揃っている場合だけ追加します。元データのディレクトリ名が `cherry` や `vivi` でも、公開するときの話者 ID には `vtuber_` を付けます。データセットの利用権限と各配布元の注意書きは学習と公開の前に確認してください。ゲーム由来の音声は研究・非商用を前提としたものがあり、LoRA のファイルに付けるライセンスだけで元音声の権利が変わるわけではありません。

## 学習前の確認

1. 対象モデルのチェックポイントに埋め込まれた構成と、Large 用 LoRA 設定の `model` 欄が一致することを確認します。LoRA の学習項目は `configs/train_v4_small_lora.yaml` の方式を基準とし、Large のモデル構成と duration predictor の値だけを正しく置き換えます。
2. 既存の `data/<speaker>/manifest.jsonl` と `latents/` が揃っている場合は再抽出しません。足りない話者だけ元データから作り、音声と書き起こしの対応、および manifest が参照する latent ファイルの存在を確認します。途中で止まった前処理の短い manifest を完成品として扱わないでください。
3. クラスタではホスト上で学習プログラムや仮想環境を実行せず、ソースを配置して Docker コンテナ内で依存関係の準備と学習を行います。GPU は他の利用者の処理を調べ、空いているものだけを割り当てます。
4. 学習用の `pyproject.toml` と `uv.lock` が Atmos Python SDK v0.1.6 以降に対応していることを確認します。イメージだけを更新しても、コンテナ起動時の `uv sync --frozen --extra atmos` が古い固定値を再インストールする可能性があります。コンテナ内の実際の SDK 版を確認してください。
5. `ATMOS_TOKEN` と Atmos の接続先をコンテナへ渡し、トークンだけで非公開プロジェクトと `/api/projects/<project_id>/jobs` を取得できることを確認します。ジョブ一覧が空でも、HTTP 200 と正しい `items` 配列が返るかを見ます。メトリクスの送信先は Atmos とし、接続できないときに無言で送信を無効にしないでください。

## 実行と再開

最初は1話者だけで、学習の最初の step と `train/loss` の Atmos 到着、GPU メモリ、定期 checkpoint の保存を確認します。出力は `outputs_v4_large/<speaker>_lora/` のように世代を分け、Small の `outputs/<speaker>_lora/` を上書きしません。各話者の Atmos 名も世代が分かる名前にします。

`train.py` を使うときは、Large 用設定、対応する manifest、Large のベースチェックポイント、出力先を明示します。`--metrics-backend atmos`、`--metrics-project`、`--metrics-run-name` を渡します。中断した学習は、保存済みの `checkpoint_<step>` を `--resume` に指定し、同じ出力先と同じ Large ベースを `--init-checkpoint` に渡して続けます。checkpoint がない場合は再開できません。既存の出力や途中まで作ったデータを削除してやり直さないでください。

複数GPU・複数ノードへ広げるのは、最初の話者で checkpoint とメトリクスを確認した後です。`stream_pipeline.sh` と `train_multi_speaker.sh` の既定値は Small 用です。Large にそのまま流用すると、ベースモデル、出力先、Atmos 名、学習 step 数が意図と異なり、失敗した話者を完了扱いにする場合もあります。担当話者、モデル世代、出力先、失敗状態を区別できる実行方法を使い、失敗を完了数に含めないでください。Atmos 上で `finished` と表示されただけで成果物が揃ったと判断せず、プロセスの終了状態と checkpoint も確認します。

validation の推移と生成サンプルを見て採用する checkpoint を選びます。`scripts/lora/export_lora_to_safetensors.py` で一つの話者 LoRA に書き出し、Large ベースとの互換性を確認した後、[ultemica/irodori-tts](https://huggingface.co/ultemica/irodori-tts) の `v4-large/<category>/<speaker>.safetensors` に保存します。`v4.1-small/` 以下は変更しません。アップロード後は対象 ID とファイル数を話者一覧に突き合わせ、未学習、失敗、検証済みを区別して記録します。
