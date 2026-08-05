# Stat Arb JP Equities User Guide（日本語）

---

# 1. このシステムでできること

このシステムは，日本株を対象とした Statistical Arbitrage（Pairs / Multi-asset Statistical Arbitrage）の研究・検証・実運用を目的としている．

現在実装されている機能は以下．

* 株価データ取得（Yahoo Finance）
* 前処理
* PCA・クラスタリングによる候補生成
* Johansen共和分検定
* スプレッド生成
* スプレッド評価
* ランキング
* シグナル生成
* バックテスト
* IBKR / Alpaca Paper Trading

処理の流れは

```
価格データ
    ↓
前処理
    ↓
PCA・クラスタリング
    ↓
Johansen
    ↓
スプレッド評価
    ↓
ランキング
    ↓
シグナル生成
    ↓
バックテスト
    ↓
Paper Trading
```

---

# 2. 実験条件の変更

実験条件は

```
src/statarb/config/settings.py
```

のみを編集する．

主な設定項目

| 設定                      | 意味                           |
| ----------------------- | ---------------------------- |
| UNIVERSE                | TOPIX100 / TOPIX500          |
| START_DATE              | 開始日                          |
| END_DATE                | 終了日                          |
| N_CLUSTERS              | クラスタ数                        |
| MIN_CLUSTER_SIZE        | 最小クラスタサイズ                    |
| MAX_CLUSTER_SIZE        | 最大クラスタサイズ                    |
| MIN_ASSETS              | Johansen対象の最小資産数             |
| MAX_ASSETS              | Johansen対象の最大資産数             |
| TOP_N_SPREADS           | 採用スプレッド数                     |
| ROLLING_WINDOW          | Rolling Z-score期間            |
| ENTRY_THRESHOLD         | エントリー閾値                      |
| EXIT_THRESHOLD          | イグジット閾値                      |
| THRESHOLD_METHOD        | fixed / gaussian / empirical |
| SPREAD_METHOD           | static / kalman              |
| BROKER                  | IBKR / ALPACA                |
| PAPER_TRADING           | Paper Trading利用可否            |
| TOP_N_EXECUTION_SPREADS | 実際に注文するスプレッド数                |

---

# 3. 研究パイプライン実行

プロジェクトルートで

```bash
uv run python -m statarb.pipeline
```

を実行する．

実行後，

```
Research pipeline finished
```

と表示される．

その後，

```
results/
```

に結果が保存される．

---

# 4. 出力ファイル

| ファイル                  | 内容            |
| --------------------- | ------------- |
| summary_fixed.csv     | 固定閾値          |
| summary_gaussian.csv  | Gaussian閾値    |
| summary_empirical.csv | Empirical閾値   |
| summary_kalman.csv    | Kalman Filter |
| selected_spreads.pkl  | 採用スプレッド       |

---

# 5. バックテスト結果の読み方

主要指標

* Total Return
* Annualized Return
* Volatility
* Sharpe Ratio
* Max Drawdown
* Win Rate
* Number of Trades

確認する順番

1. Sharpe Ratio
2. Max Drawdown
3. CAGR（Annualized Return）
4. Win Rate
5. Number of Trades

---

# 6. Threshold の違い

## fixed

固定閾値

```
Entry = 2σ
Exit = 0.5σ
```

## gaussian

理論上最適な閾値

## empirical

実データから最適化した閾値

---

# 7. Spread の違い

## static

Johansen β を固定

## kalman

β を時変化

---

# 8. Paper Trading

## IBKR

### 起動

Trader Workstation を起動

Paper Account でログイン

API を有効化

### 実行

```bash
uv run python -m statarb.pipeline
```

または

```python
IBKRExecution.submit_order(...)
```

### 確認

Paper Portfolio

Open Orders

Activity

---

## Alpaca

Paper Account を利用

`.env`

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

設定後，

```python
AlpacaExecution.submit_order(...)
```

---

# 9. よく使うコマンド

研究

```bash
uv run python -m statarb.pipeline
```

テスト

```bash
uv run pytest
```

Ruff

```bash
uv run ruff check .
```

Format

```bash
uv run ruff format .
```

---

# 10. 開発フロー

通常は

```
config/settings.py
      ↓
pipeline実行
      ↓
summary確認
      ↓
selected_spreads確認
      ↓
必要ならパラメータ変更
      ↓
再実行
```

を繰り返す．

---

# 11. 実運用前チェックリスト

* Universe確認
* 日付確認
* Threshold確認
* Spread Method確認
* Broker確認
* Paper Trading確認
* Summary確認
* Selected Spread確認
* Paper Tradingで注文確認
* 問題なければ実運用
