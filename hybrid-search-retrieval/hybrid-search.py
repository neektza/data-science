import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datasets import load_dataset
    import polars as pl
    return (load_dataset,)


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    return (SentenceTransformer,)


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _(load_dataset):
    dataset = load_dataset("ms_marco", "v1.1")
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    dataset['train'][55]
    return


@app.cell
def _(dataset):
    query = dataset['train'][55]['query']
    return (query,)


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return (model,)


@app.cell
def _(model, query):
    query_enc = model.encode(query)
    return (query_enc,)


@app.cell
def _(query_enc):
    query_enc
    return


@app.cell
def _(dataset):
    passages = dataset['train'][55]['passages']['passage_text']
    return (passages,)


@app.cell
def _(passages):
    passages
    return


@app.cell
def _(model, passages):
    psgs_enc = model.encode(passages)
    return (psgs_enc,)


@app.cell
def _(psgs_enc):
    psgs_enc
    return


@app.cell
def _(psgs_enc, query_enc, torch):
    similarity = torch.cosine_similarity(torch.tensor(query_enc), torch.tensor(psgs_enc[0]), dim=0)
    return (similarity,)


@app.cell
def _(similarity):
    similarity
    return


if __name__ == "__main__":
    app.run()
