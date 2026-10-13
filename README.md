# EllipseForegroundのサイト

このブランチは、`https://lib.routersys.com/EllipseForeground/`で公開するサイトの置き場です。公開元は`docs`で、GitHub Pagesの公開元に指定して使います。構成は、R128NetとWorldNetのサイトに合わせました。入口のページは英語と日本語の2つです。デモにあたるページには、論文のPDFビューアを置いています。

## 構成

| 場所 | 内容 |
|---|---|
| `docs/index.html`、`docs/ja/index.html` | 入口のページ。英語と日本語 |
| `docs/paper/`、`docs/ja/paper/` | 論文のPDFビューア。英語と日本語 |
| `docs/paper/viewer.js`、`docs/paper/viewer.css` | ビューアの本体。2つの言語のページが共有します |
| `docs/paper/ellipse-foreground-extraction.pdf` | 論文のPDF。mainブランチの`docs`にあるPDFの写しです |
| `docs/paper/vendor/pdfjs` | PDF.js 6.3.289（Apache-2.0）。`SOURCE.txt`に出どころとハッシュを書いてあります |
| `docs/404.html`、`docs/sitemap.xml`、`docs/icon.png`、`docs/.nojekyll` | 見つからないときのページ、サイトマップ、アイコン、Jekyllを使わない印 |
| `check` | ローカルでの検証 |

ビューアは、PDF.jsでページを描き、本文の層を重ねます。表示するのは見えている付近のページだけです。ページの移動、拡大縮小、本文の検索ができ、`#page=12`のようにページを指定して開けます。PDFは外部へ送りません。

## ローカルで確かめる

サイトを、公開先と同じ`/EllipseForeground/`の下に置いて動かします。

```console
node check/serve.mjs
```

表示されたURL（既定は`http://127.0.0.1:18123/EllipseForeground/`）をブラウザーで開いてください。ポートは、`node check/serve.mjs 8080`のように変えられます。`file://`で開くと、ブラウザーがワーカーの読み込みを許さないので、PDFは表示されません。

検査は2段あります。

```console
node check/check.mjs
```

1段目は、ブラウザーを使わない静的な検査です。ページごとのcanonical、hreflang、Open Graph、JSON-LD、内部リンクの行き先、サイトマップとの一致、ビューアが使う要素とメッセージの有無、日本語のページの和欧文の間の空白、PDFがmainブランチのPDFと同じであること、PDF.jsのハッシュを確かめます。

```console
cd check
npm ci
node browser.mjs
```

2段目は、ヘッドレスのChromeでの検査です。Chromeの場所は、環境変数`CHROME_PATH`で指定できます。確かめるのは、ページ数、描画、本文の層、移動、拡大縮小、検索、`#page=12`での開き方、幅390画素での表示、入口のページと404です。画面の写真は`check/out`に書き出されます。

## PDFを更新するとき

mainブランチの`docs`のPDFを差し替えたら、同じファイルを`docs/paper`へコピーし、`docs/sitemap.xml`の`lastmod`を直して、静的な検査を通してください。PDFがmainブランチのものと違うと、検査が失敗します。

## ライセンス

ビューアとページは、リポジトリの[BSD 3-Clause License](https://github.com/routersys/EllipseForeground/blob/main/LICENSE.txt)です。PDF.jsは、Apache License 2.0です。全文は`docs/paper/vendor/pdfjs/LICENSE.txt`にあります。論文のPDFにある立ち絵1と立ち絵2の図には、キャラクターのデータの[利用条件](https://github.com/routersys/EllipseForeground/blob/main/data/own/LICENSE.txt)が適用されます。
