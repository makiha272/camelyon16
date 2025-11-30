# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import slideflow as sf
    import pandas as pd
    return pd, sf


@app.cell
def _(sf):
    P = sf.create_project(
      root='/workspace/slideflow_project',
      annotations="./annotations.csv",
      slides='/workspace/dataset/camelyon16/images/'
    )
    return


@app.cell
def _(pd):
    #CSVを読み込み
    df = pd.read_csv("/workspace/slideflow_project/annotations.csv")

    # patient もしくは slide の値からカテゴリー抽出
    def extract_category(value: str):
        # 接頭辞が test_ / normal_ / tumor_ のいずれかなら、それをそのまま返す
        if value.startswith("test_"):
            return "test"
        elif value.startswith("normal_"):
            return "normal"
        elif value.startswith("tumor_"):
            return "tumor"
        else:
            return None   # 不明な場合

    # category列を埋める
    df["category"] = df["patient"].apply(extract_category)
    return (df,)


@app.cell
def _(df):
    # sample列を作成: category が normal/tumor の先頭10件を True、それ以外は False
    df['sample'] = False
    for cat in ['normal', 'tumor']:
        mask = df['category'] == cat
        idxs = df[mask].index[:10]
        df.loc[idxs, 'sample'] = True
    return


@app.cell
def _(df):
    # 確認: カテゴリごとの True 数とサンプル行の表示
    print('sample counts per category:')
    print(df.groupby('category')['sample'].sum())
    df
    return


@app.cell
def _(df):
    df.to_csv("/workspace/slideflow_project/annotations.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
