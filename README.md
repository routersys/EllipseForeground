# EllipseForeground

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.11%2B-orange.svg)](#)
[![Release](https://img.shields.io/github/v/release/routersys/EllipseForeground.svg)](https://github.com/routersys/EllipseForeground/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23318600.svg)](https://doi.org/10.5281/zenodo.23318600)

---

楕円で大まかに指定した範囲から、画像の前景のアルファマットを求めるPythonのライブラリです。
論文「楕円による大まかな指定からの前景抽出」の参照実装で、評価のコードと論文の原稿も収録しています。
単連結法のデンドログラムの上で、楕円に由来する事前分布と画素ごとの手がかりの和を最大にする切断を、動的計画法で求めます。
学習済みのモデルや追加の注釈は使わず、入力は画像と楕円だけです。

論文は、[ブラウザーで読めます](https://lib.routersys.com/EllipseForeground/paper/)。PDFは外部へ送りません。

---

## 目次

1. [概要](#概要)
2. [動作要件](#動作要件)
3. [インストール方法](#インストール方法)
4. [主な機能](#主な機能)
   - [1. アルファマットを求める](#1-アルファマットを求める)
   - [2. 評価を再現する](#2-評価を再現する)
   - [3. 基準値と照合する](#3-基準値と照合する)
   - [4. 論文を組む](#4-論文を組む)
   - [5. 計算時間を測る](#5-計算時間を測る)
5. [リポジトリの構成](#リポジトリの構成)
6. [画像と素材](#画像と素材)
7. [制限事項](#制限事項)
8. [注意事項](#注意事項)
9. [不具合の報告](#不具合の報告)
10. [免責事項](#免責事項)
11. [サードパーティライセンス](#サードパーティライセンス)
12. [ライセンス](#ライセンス)
13. [引用](#引用)

---

## 概要

本ライブラリは、画像と楕円を受け取り、楕円が指す前景のアルファマットを返します。

処理の流れは次のとおりです。画像を幅256画素に縮め、隣り合う画素を色の差の小さい順に併合して、デンドログラム（階層的な木）を作ります。一方、楕円からは事前分布が、色、テクスチャ、パッチの三つの手がかりからは画素ごとのスコアが求まります。この二つに境界項を加えた和を最大にする切断を、木の上の動的計画法で厳密に選ぶのが中心の処理です。選んだ領域を、縮める前の画像の大きさに合わせて整えたものが、返すアルファマットです。

手法の詳細と評価の結果は、論文に書いてあります。PDFは[`docs/ellipse-foreground-extraction.pdf`](docs/ellipse-foreground-extraction.pdf)にあり、原稿は`paper/source`にあります。設定のデフォルト値は、論文の付録Aに挙げた定数です。

---

## 動作要件

| 項目 | 要件 |
|---|---|
| Python | 3.11以降 |
| 必須のライブラリ | NumPy 2.0以降、Pillow 10.0以降、SciPy 1.11以降 |
| 図の生成 | Matplotlib 3.8以降。追加の依存`figures`で入ります |
| 論文のPDF | Google Chrome。ヘッドレスで実行します |
| 計算時間の測定 | .NET 10のSDK |

※ 開発はWindows 11で、継続的なテストはUbuntu上のPython 3.11と3.13で行っています。macOSでは確かめていません。

---

## インストール方法

ライブラリだけを使うときは、リリースのwheelを入れます。評価、論文、基準値との照合を使うときは、ソースから入れます。

### wheelから入れる

1. [Releases](https://github.com/routersys/EllipseForeground/releases/latest)ページから最新のパッケージ（`EllipseForeground_vX.Y.Z.zip`）をダウンロードします。
2. 解凍すると出てくるwheel（`ellipse_foreground-X.Y.Z-py3-none-any.whl`）を、次のコマンドでインストールします。

```console
python -m pip install ellipse_foreground-X.Y.Z-py3-none-any.whl
```

wheelに入っているのは、`ellipse_foreground`のパッケージだけです。`harness`、`paper`、`data`、`results`は入っていません。

### ソースから入れる

1. リポジトリをクローンします。
2. リポジトリの直下で、次のコマンドを実行します。
3. `python -m pytest`でテストが通ることを確かめます。

```console
git clone https://github.com/routersys/EllipseForeground.git
cd EllipseForeground
python -m pip install -e ".[dev,figures]"
```

評価と論文のコマンドは、リポジトリの直下にある`data`と`results`をデフォルトの場所として読み書きします。編集可能なインストールで使ってください。

---

## 主な機能

### 1. アルファマットを求める

`Analysis`に画像を渡し、`extract`に楕円を渡します。

```python
import numpy as np
from PIL import Image

from ellipse_foreground import Analysis, Ellipse, Mix

source = np.asarray(Image.open("input.png").convert("RGB"))
analysis = Analysis(source.astype(np.float32) / 255.0, source)
ellipse = Ellipse(center_x=640, center_y=330, radius_x=190, radius_y=300)
alpha = analysis.extract(ellipse, Mix.ADAPTIVE_COLOR)
Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).save("alpha.png")
```

`Analysis`の第1引数は、0から1の値を持つfloat32のRGB配列です。第2引数は同じ画像の8ビットの配列で、テクスチャの手がかりの計算に使われます。テクスチャの平坦さは縮小すると失われるので、第2引数には縮小していない画像を渡してください。`Ellipse`の座標と半径は、第1引数の画像の画素で表します。`extract`が返すのは、第1引数と同じ高さと幅のfloat32の配列で、値の範囲は0から1です。

同じ画像で楕円だけを変えるときは、`Analysis`を作り直さず、`extract`を呼び続けてください。木と境界項の計算が、画像ごとに1回で済むためです。

手がかりの組み合わせは`extract`の第2引数で選び、省略したときは`Mix.ADAPTIVE_COLOR`が使われます。

| 値 | 内容 |
|---|---|
| `Mix.PRIOR_ONLY` | 楕円に由来する事前分布だけを使います |
| `Mix.TEXTURE_ONLY` | 事前分布にテクスチャの手がかりを加えます |
| `Mix.APPEARANCE_ONLY` | 事前分布に色とパッチの手がかりを加えます |
| `Mix.FIXED_MIX` | 三つの手がかりを固定の重みで併用します |
| `Mix.ADAPTIVE_COLOR` | 色の重みを画像ごとに決めて、三つの手がかりを併用します |
| `Mix.ADAPTIVE_COLOR_PATCH` | 色とパッチの重みを画像ごとに決めて、三つの手がかりを併用します |

設定は`Settings`で変えます。次の例は、項目ごとのデータクラスを渡して、縮小する幅を128画素にしたものです。

```python
from ellipse_foreground import Analysis, Settings
from ellipse_foreground.settings import WorkSettings

analysis = Analysis(image, source, Settings(work=WorkSettings(width=128)))
```

1280×640画素の画像では、`Analysis`の構築に約0.3秒、1回の`extract`に約1秒かかります。Intel Core i7-1360P、Python 3.13、NumPy 2.5.4で、ほかの処理を止めて測りました。`Analysis`は5つの場面の中央値で0.26〜0.41秒、`extract`は5つの場面と各3つの楕円の15回の中央値で0.8〜1.4秒の範囲でした。各回は2回測って速い方を採っています。最初の`extract`は、縁の仕上げの準備のために、約0.5秒長くかかります。このマシンは負荷や温度で速度が2割ほど変わるため、2回の測定の中央値は0.93秒と1.14秒でした。

### 2. 評価を再現する

論文の評価は、`python -m ellipse_foreground.evaluation`にデータセットの名前を1つ渡すと再現できます。

```console
python -m ellipse_foreground.evaluation sprite1
```

| データセット | 内容 | 場面の数 |
|---|---|---|
| `sprite1` | 立ち絵1。星雲の背景で色転写の強さを5段階に変えた条件と、ほかの7背景の条件 | 12 |
| `sprite2` | 立ち絵2。8種の背景 | 8 |
| `public0` | 公開素材の立ち絵を、色転写なしで16種の背景に合成した場面 | 102 |
| `public100` | 同じ場面に、強さ100%の色転写をかけたもの | 102 |
| `collection0`、`collection100` | 手元の素材から作る場面。色転写の強さは0%と100% | 素材の数の6倍 |
| `photo` | 手元の写真と、手作業のアルファマットの組。PPM-100を想定している | 写真の枚数 |

結果は`results`の下に書かれます。1行が1試行のJSONLファイルで、フォルダーは実験の種類ごとに分かれていて、`comparison`が構成の比較、`ablation`が境界項や仕上げなどの条件の変更、`models`が楕円の誤差モデルの変更、`resolution`が縮小する幅の変更です。途中で止めたときは、同じコマンドをもう一度実行すれば、終わった場面を飛ばして続きから実行されます。

| オプション | 内容 |
|---|---|
| `--experiments` | 実行する実験をコンマで区切って指定します。`trials`、`comparison`、`models`、`resolution`から選べます。`comparison`は`trials`のうち構成の比較だけです。省略すると、`comparison`を除く三つを実行します |
| `--start`、`--end` | 場面の番号の範囲を指定します。範囲を指定したときは、結果が範囲を付けた名前のファイルに書かれます |
| `--data` | `data`の場所です。デフォルトは、リポジトリの`data`です |
| `--results` | 結果の保存先です。デフォルトは、リポジトリの`results`です |
| `--sprites` | 手元のデータセットの立ち絵です。画像を置いたフォルダーか、1行に1つの画像のパスを書いたテキストファイルを渡します。フォルダーでは、png、webp、jpg、jpegを名前の順に読みます |
| `--backgrounds` | 手元のデータセットの背景です。書き方は`--sprites`と同じです |
| `--photos` | `photo`で使う写真のフォルダーです。`image`に写真、`matte`に同じ名前のアルファマットを置きます |
| `--seed` | 手元のデータセットの場面を決める乱数の種です。デフォルトは78です |
| `--grounded` | 手元のデータセットで、立ち絵を画像の下端に接する位置に置きます |

手元のデータセットの場面の名前は、`sprite00_bg5_3`のように、立ち絵の番号、背景の番号、作った順の番号で表します。写真の場面の名前は`photo000`のように、名前の順の番号で表します。画像のファイル名は結果に残りません。

写真は、幅が1280画素を超えるものを面積平均で幅1280画素に縮め、それ以外は元の大きさのまま使います。色転写はかけず、楕円の誤差モデルと評価の指標は、ほかのデータセットと同じです。写真は再配布の条件が個別に定まっているため、リポジトリには含めません。

公開素材の1データセットは、Intel Core i7-1360Pでほかの評価を並行して実行したときに、約3.5時間かかりました。

GrabCutとSAM（ViT-B）を比較の基準にする評価は、依存を入れてから別のコマンドで実行します。入力は、評価の楕円の外接矩形です。

```console
pip install -e ".[baselines]"
python -m ellipse_foreground.evaluation.baselines public0 --checkpoint downloads/sam/sam_vit_b_01ec64.pth
```

重みは、Metaが配布する`sam_vit_b_01ec64.pth`（約375MB）を利用者が取得します。リポジトリには含めません。`--checkpoint`を省くとGrabCutだけを実行します。`photo`には`--photos`が必要です。結果は`results/baselines`に書かれます。

### 3. 基準値と照合する

`harness`は、論文の場面から選んだ画像で各段の出力を作り、出力のハッシュを`harness/golden.json`の基準値と比べます。コードを変えても出力が変わっていないことを確かめるために使います。ゴールデンファイルテストと呼ばれる手法です。

```console
python -m harness --verify
```

| 指定 | 内容 |
|---|---|
| なし、または出力先 | 各ケースを描き、PNGを出力先に書き、ハッシュを表示します。出力先のデフォルトは`harness-output`です |
| `--golden` | 基準値を`harness/golden.json`に書きます |
| `--verify` | 基準値と照合します。一致しないケースがあると、終了コードが1になります |
| `--transition` | 同じ解析を使い回した結果が、解析を作り直した結果と一致することを確かめます |
| `--compare <前> <後>` | 二つの出力先のPNGを、画素ごとに比べます |
| `--benchmark` | 段ごとの処理時間を測ります |

`python -m pytest -m golden`で、同じ照合をテストとして実行できます。`python -m pytest`だけでは実行しません。

### 4. 論文を組む

```console
python -m paper build
```

`results`と`data`を読み、図、表、本文を作り直して、`paper/build`に`paper.html`と`paper.pdf`を書き出します。表と本文の結果の数値は、すべて`results`と`data`が元です。リポジトリの`docs/ellipse-foreground-extraction.pdf`は、開発者がこのコマンドの結果で更新します。

| オプション | 内容 |
|---|---|
| `--results`、`--data`、`--timing` | 結果、画像、計算時間の測定結果の場所です |
| `--output` | 書き出し先です。デフォルトは`paper/build`です |
| `--chrome` | Google Chromeの実行ファイルです。デフォルトはWindowsの`C:\Program Files\Google\Chrome\Application\chrome.exe`です |
| `--no-figures` | 図を作り直しません |
| `--no-pdf` | PDFを作りません |

### 5. 計算時間を測る

```console
python -m paper timing
```

5場面のスコアを、木の動的計画法と、画素グラフの最小カット（Dinic法）で解いたときの、1回あたりの時間が`results/timing.json`に書かれます。どちらもC#で書いた`paper/solvers`のプログラムで、単一スレッドです。実行の最初に、`dotnet build`で2つのプログラムを作ります。測るのは、木とグラフの構築と最初の実行を除いた、2回目の実行の時間です。ほかの重い処理は止めてから実行してください。

PyMaxflowを入れてある場合（`pip install -e ".[timing]"`）は、同じ問題をBoykovとKolmogorovのアルゴリズムでも解き、`maxflow`の呼び出しだけの時間を`results/timing.json`に加えます。グラフの構築は測りません。PyMaxflowはGPLで配布されているため、依存には含めず、入れたときだけ使います。

---

## リポジトリの構成

| 場所 | 内容 |
|---|---|
| `src/ellipse_foreground` | 参照実装。木、切断、手がかり、境界項、精緻化、画素単位のグラフカット |
| `src/ellipse_foreground/evaluation` | 評価。場面の作成、楕円の誤差モデル、指標、実験の実行 |
| `harness` | 基準値との照合 |
| `paper` | 論文の原稿、表と図の作成、組版、計算時間の測定 |
| `docs` | 論文のPDF |
| `results` | 評価の結果と、計算時間の測定結果 |
| `data` | 評価に使う画像と、立ち絵のデータ |
| `tests` | テスト |

---

## 画像と素材

`data`の画像のライセンスは、場所ごとに次のとおりです。

| 場所 | 内容 | ライセンス | 作者と出典 |
|---|---|---|---|
| `data/own` | 立ち絵1と立ち絵2の前景の色とアルファマット | [独自の利用条件](data/own/LICENSE.txt)。AIの学習への利用は禁止です | routersys |
| `data/sprites`の`breezy_*` | 立ち絵5枚 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.ja) | Breezy氏の「Visual Novel Sprites!」 https://breezy-the-cat.itch.io/visual-novel-sprites |
| `data/sprites`の`cyanea*`（`cyanea8_casual`を除く） | 立ち絵11枚 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja) | 4cher氏（Cyanea）の「[FREE-TO-USE] Visual Novel Sprites」のセット1、2、4、5、6、7。各セットのページは https://iam-cyanea.itch.io/set1-vnsprites のように、末尾が`set`と番号と`-vnsprites`です |
| `data/sprites`の`cyanea8_casual` | 立ち絵1枚 | ライセンスの表示なし。作者の説明文の条件による | 4cher氏（Cyanea）の同じ題名のセット8 https://iam-cyanea.itch.io/set8-vnsprites |
| `data/backgrounds` | 背景16枚 | CC0 1.0 | 作者と出典のページは`data/backgrounds/manifest.json`に書いてあります |

セット8の販売ページには、ライセンスの表示がありません。説明文は、商用利用を許し、クレジットを不要とし、自作と主張することと販売することをしないよう求めています。再配布についての記述はありません。セット1、2、4、6、7の説明文も、素材を販売しないよう求めています。Breezy氏のページには、素材を生成AIに使わないでほしいという依頼と、使ったときは知らせてほしいという依頼があります。

立ち絵1と立ち絵2は、同じキャラクターで髪型だけが異なります。`data/own`にあるのは、動画編集ソフトの書き出しから復元した、キャラクターのアルファマットと前景の色だけです。背景の画素は含まず、書き出しそのものは、背景を再配布できないため収録していません。評価の場面の背景は、`data/backgrounds`のCC0の画像です。

立ち絵1と立ち絵2のデータは、クリエイティブ・コモンズなどの定型のライセンスではなく、`data/own/LICENSE.txt`の条件で提供します。コードを動かした論文の結果の再現、個人の勉強と非営利の研究での評価、結果の図への掲載は許可します。AIの学習への利用、営利を目的とする利用、データだけの配布は禁止です。論文のPDFと原稿に載る立ち絵1と立ち絵2の図にも、同じ条件が適用されます。

書き出しから作り直すときは、`python -m ellipse_foreground.evaluation.export sprite1 <フォルダー>`を使います。フォルダーには、背景つきの通常の出力`composite.png`、前景だけの出力`foreground_only.png`、背景だけの出力`background_only.png`と、元の色の出力（立ち絵1は`original_colors_composite.png`、立ち絵2は`original_colors_foreground.png`）を置きます。

`data/sprites`の画像は、配布元のファイルの内容を変えず、ファイル名だけを変えて収録しました。`data/backgrounds`の画像は、配布元の画像を16対9に切り出し、1920×1080画素に拡大縮小して、JPEG（品質95）で保存したものです。Wikimedia Commonsの12枚は、Commonsが生成する幅1920画素の画像から作りました。各フォルダーの`manifest.json`には、作者、題名、ライセンス、出典、収録したファイルのSHA-256を記録してあります。

論文の評価に使った17体の立ち絵は、第三者の著作物のため、リポジトリに含めていません。`results`にあるのは、その評価の数値だけです。

---

## 制限事項

- 評価は、立ち絵と背景を合成した画像だけで行いました。写真の人物は評価していません。
- 学習済みの分割モデルとの比較はしていません。
- 処理は、CPU上のNumPyで書いてあります。1280×640画素の画像で1回の`extract`に約1秒かかるため、楕円を動かしながら対話的に使うには遅いです。
- 前景が背景と同じ色の一様な塗りになる画像では、三つの手がかりがすべて働きません。論文の17体の転写100%には、背景がほぼ一様な壁紙で、強さ100%の色転写により前景が背景と同じ色になる場面が6場面あります。その12試行では、三つの手がかりの楕円内のAUCがすべて0.5で、手がかりを使う構成のIoUの平均は0でした。

---

## 注意事項

- 画像の大きさの上限: `--sprites`、`--backgrounds`、`--photos`を使う手元のデータセットでは、Pillowの画像の大きさの上限の検査を外して画像を開きます。信頼できない画像を渡さないでください。
- 基準値の環境: `harness/golden.json`は、Windows、Python 3.13.1、NumPy 2.5.4、SciPy 1.18.1、Pillow 12.3.0で書いた基準値です。環境が違うと、出力のハッシュが一致しないことがあります。`--verify`は、環境が違うときにその旨を警告します。
- メモリ: 評価は、1プロセスで約1.2GBを確保します。測定は、Windows 11で、縮小する幅が256画素のときです。複数のプロセスを並行して実行するときは、メモリの空きを確かめてください。評価を5本並行し、ほかの重い処理も動かしたところ、32GBのメモリで、メモリ不足で止まったことがあります。
- 図のフォント: 図の日本語は、BIZ UDPGothicで描きます。入っていない環境では、日本語が表示されません。
- 帰属の表示: `data/sprites`のCyaneaの画像のうち、CC BY 4.0のもの（`cyanea8_casual`以外）を使うときは、作者名、ライセンスの名前、ライセンスへのリンクを示してください。画像を変更したときは、変更したことも示してください。立ち絵1と立ち絵2のデータを結果の図に載せるときは、`data/own/LICENSE.txt`に従って、著作者のroutersysとリポジトリのURLを示してください。

---

## 不具合の報告

不具合はGitHubの[Issues](https://github.com/routersys/EllipseForeground/issues)へ報告してください。本ライブラリは、不具合の内容を自動で送信しません。脆弱性は、[セキュリティポリシー](SECURITY.md)に従って非公開で報告してください。

報告には次の内容を書いてください。

| 項目 | 内容 |
|---|---|
| 環境 | Python、NumPy、SciPy、Pillowのバージョンと、OS |
| 再現手順 | 何をどの順に行ったか。使った画像と楕円、`Settings`を変えたときはその内容 |
| 結果 | 期待した動作と、実際に起きたこと |
| 出力 | エラーの内容と、スタックトレース。省略せずに貼ってください |

画像は、公開してよいものだけを添付してください。

---

## 免責事項

本ライブラリは BSD 3-Clause License のもとで公開されています。

本ソフトウェアは「現状のまま」提供されており、明示・黙示を問わず、商品性、特定目的への適合性、および権利非侵害に関する保証を含む、いかなる種類の保証も行いません。

作者は、本ライブラリの使用または使用不能に起因するいかなる損害についても、一切の責任を負いません。
ご利用は自己責任でお願いします。

---

## サードパーティライセンス

本ライブラリは、次のサードパーティの素材を同梱しています。ライセンスの全文の収録先は、リポジトリの[`.github/LICENSE`](.github/LICENSE)です。

| 素材 | 用途 | ライセンス | 著作権表示 |
|---|---|---|---|
| Breezy氏の「Visual Novel Sprites!」 | 評価に使う立ち絵 | CC0 1.0 | 著作権表示は不要です |
| 4cher氏（Cyanea）の「[FREE-TO-USE] Visual Novel Sprites」のセット1、2、4、5、6、7 | 評価に使う立ち絵 | CC BY 4.0 | 4cher (Cyanea) |
| 4cher氏（Cyanea）の「[FREE-TO-USE] Visual Novel Sprites」のセット8 | 評価に使う立ち絵 | ライセンスの表示なし。作者の説明文の条件による | 4cher (Cyanea) |
| 背景16枚 | 評価に使う背景 | CC0 1.0 | 著作権表示は不要です |

NumPy、SciPy、Pillow、Matplotlibは同梱していません。インストールの際には、それぞれのライセンスが適用されます。

行動規範は、[Contributor Covenant](https://www.contributor-covenant.org)のバージョン2.0を基にしたものです。

---

## ライセンス

コード、テスト、論文の原稿のライセンスは、[BSD 3-Clause License](LICENSE.txt)です。`data`の画像は、[画像と素材](#画像と素材)の表のライセンスに従います。ただし、論文のPDFと原稿に載る立ち絵1と立ち絵2の図は、`data/own/LICENSE.txt`の条件に従います。

---

## 引用

このリポジトリを引用するときは、[`CITATION.cff`](CITATION.cff)を使ってください。GitHubの「Cite this repository」からも取れます。

リポジトリはZenodoに登録してあります。すべてのバージョンを表すDOIは[10.5281/zenodo.23318600](https://doi.org/10.5281/zenodo.23318600)で、v1.0.0のDOIは[10.5281/zenodo.23318601](https://doi.org/10.5281/zenodo.23318601)、v1.1.0のDOIは[10.5281/zenodo.23330045](https://doi.org/10.5281/zenodo.23330045)、v1.1.1のDOIは[10.5281/zenodo.23337765](https://doi.org/10.5281/zenodo.23337765)、v1.2.0のDOIは[10.5281/zenodo.23343361](https://doi.org/10.5281/zenodo.23343361)です。
