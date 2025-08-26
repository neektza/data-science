import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    train_df = pl.read_csv("food-demand-forecasting/data/train.csv")
    return (train_df,)


@app.cell
def _(train_df):
    train_df
    return


if __name__ == "__main__":
    app.run()
