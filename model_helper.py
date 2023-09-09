import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import merlin.io
import merlin.models.tf as mm
import numpy as np
import tensorflow as tf
from merlin.models.tf.core.encoder import TopKEncoder
from merlin.models.tf.outputs.base import DotProduct
from merlin.models.tf.transforms.bias import PopularityLogitsCorrection
from merlin.models.utils import schema_utils
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema import ColumnSchema, Schema, Tags
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers


warnings.filterwarnings("ignore")


def topk_metrics_aggregator(top_ks: list = [100, 10]):
    return [
        tf.keras.metrics.AUC(from_logits=True, name="auc"),
        mm.TopKMetricsAggregator.default_metrics(top_ks=top_ks),
    ]


def _build_contrastive_output(
        data,
        logq_sampling_correction=False,
        logits_temperature=0.1,
        negative_samplers=["in-batch"],
        store_negative_ids=True,
):
    schema = data.schema
    item_id = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    post_logits = None
    if logq_sampling_correction:
        item_id_cardinality = schema_utils.categorical_cardinalities(schema)[item_id]

        items_frequencies = data.to_ddf().compute()[item_id].value_counts().sort_index()
        # hack, needed when workflow was fit on entire dataset
        if items_frequencies.shape[0] < item_id_cardinality:
            print("[INFO] hacked the logq correction to match item cardinality")
            items_frequencies = items_frequencies.reindex(
                    range(item_id_cardinality), fill_value=0
            ).values

        post_logits = PopularityLogitsCorrection(
                items_frequencies,
                schema=schema,
        )

    return mm.ContrastiveOutput(
            DotProduct(),
            logits_temperature=logits_temperature,
            post=post_logits,
            negative_samplers=negative_samplers,
            schema=schema.select_by_tag(Tags.ITEM_ID),
            store_negative_ids=store_negative_ids,
    )


def build_towers(
        data,
        tower_dim=[(128, 64)],
        neg_sampler=["in-batch"],
        embedding_dims=None,
        logq_sampling_correction=False,
        item_categorical=None,
        regularize=None
):
    schema = data.schema
    if not neg_sampler:
        neg_sampler = ["in-batch"]

    user_tower_dim = tower_dim
    item_tower_dim = tower_dim
    if isinstance(tower_dim, dict):
        user_tower_dim = tower_dim['user']
        item_tower_dim = tower_dim['item']

    if regularize is None:
        regularize = dict(
            dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(1e-6),
        )

    # create user schema using USER tag
    user_schema = schema.select_by_tag(Tags.USER)
    # create user (query) tower input block
    user_inputs = mm.InputBlockV2(user_schema)
    # create user (query) encoder block
    query = mm.Encoder(
            user_inputs, mm.MLPBlock(user_tower_dim,
                                     no_activation_last_layer=True,
                                    **regularize
                                    )
    )

    # create item schema using ITEM tag
    item_schema = schema.select_by_tag(Tags.ITEM)
    # create item (candidate) tower input block
    item_inputs = mm.InputBlockV2(item_schema, categorical=item_categorical)
    # create item (candidate) encoder block
    candidate = mm.Encoder(
            item_inputs, mm.MLPBlock(item_tower_dim,
                                    no_activation_last_layer=True,
                                    **regularize
                                    )
    )

    def _switch_emb_dims(block, features):
        for l in block.layers[0].layers:
            if l.table.name in features:
                l.table.output_dim = features[l.table.name]

    if embedding_dims:
        if embedding_dims.get("user"):
            _switch_emb_dims(user_inputs, embedding_dims.get("user"))
        if embedding_dims.get("item"):
            _switch_emb_dims(item_inputs, embedding_dims.get("item"))

    kwargs = {
        "store_negative_ids": True,
        "logq_sampling_correction": logq_sampling_correction,
    }
    outputs = _build_contrastive_output(data, negative_samplers=neg_sampler, **kwargs)

    return mm.TwoTowerModelV2(query, candidate, outputs=outputs)


default_plot_metrics = {
    "Loss": "loss",
    "Recall@10": "recall_at_10",
    "Ndcg@10": "ndcg_at_10",
    "AUC": "auc",
}


def plot_metrics(
        train_history,
        val_history=None,
        metrics=default_plot_metrics,
        figsize=None,
        max_row=4,
):
    # Create a figure and axis
    X = math.ceil(len(metrics) / max_row)
    Y = max_row if len(metrics) > max_row else len(metrics)

    if figsize is None:
        figsize = (18, 3 * X)

    _, ax = plt.subplots(X, Y, figsize=figsize)

    for i, (k, m) in enumerate(metrics.items()):
        _ax = ax
        if X > 1:
            _ax = ax[i // max_row]
            i %= max_row
        train_metric = train_history[m]
        epochs = np.arange(1, len(train_metric) + 1)

        # Plot metric
        _ax[i].plot(epochs, train_metric, label=f"Train {k}", marker="o")
        if val_history:
            _ax[i].plot(epochs, val_history[m], label=f"Validation {k}", marker="o")
        _ax[i].set_xlabel("Epochs")
        _ax[i].set_ylabel(k)
        _ax[i].set_title(f"{k} Over Epochs")
        _ax[i].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def merge_model_history(*history):
    _history = defaultdict(list)
    for _h in history:
        h = _h.history
        for k, v in h.items():
            _history[k].extend(v)
    return _history


# hack block, as Merlin top_k_encoder is broken
def batch_predict(
        self,
        dataset: merlin.io.Dataset,
        batch_size: int,
        output_schema=None,
        index=None,
        **kwargs,
) -> merlin.io.Dataset:
    """Batched prediction using Dask.
    Parameters
    ----------
    dataset: merlin.io.Dataset
        Dataset to predict on.
    batch_size: int
        Batch size to use for prediction.
    Returns
    -------
    merlin.io.Dataset
    """

    if index:
        if isinstance(index, ColumnSchema):
            index = Schema([index])
        elif isinstance(index, str):
            index = Schema([self.schema[index]])
        elif isinstance(index, Tags):
            index = self.schema.select_by_tag(index)
        elif not isinstance(index, Schema):
            raise ValueError(f"Invalid index: {index}")

        if len(index) != 1:
            raise ValueError("Only one column can be used as index")
        index = index.first.name

    if hasattr(dataset, "schema"):
        if not set(self.schema.column_names).issubset(set(dataset.schema.column_names)):
            raise ValueError(
                    f"Model schema {self.schema.column_names} does not match dataset schema"
                    + f" {dataset.schema.column_names}"
            )

    # Check if merlin-dataset is passed
    if hasattr(dataset, "to_ddf"):
        # hack
        dataset = dataset.to_ddf().compute()

    from merlin.models.tf.utils.batch_utils import TFModelEncode

    model_encode = TFModelEncode(self, batch_size=batch_size, **kwargs)
    encode_kwargs = {}
    if output_schema:
        encode_kwargs["filter_input_columns"] = output_schema.column_names
    # hack
    predictions = model_encode(dataset, **encode_kwargs)
    if index:
        predictions = predictions.set_index(index)

    return merlin.io.Dataset(predictions)


def to_top_k_encoder(
        model,
        candidates: merlin.io.Dataset = None,
        candidate_id=Tags.ITEM_ID,
        strategy="brute-force-topk",
        k: int = 10,
        batch_size=512,
):
    output_schema = model.schema.select_by_tag(candidate_id)

    # https://github.com/NVIDIA-Merlin/models/blob/stable/merlin/models/tf/models/base.py#L2479
    candidates_embeddings = batch_predict(
            model.candidate_encoder,
            candidates,
            batch_size=batch_size,
            output_schema=output_schema,
            index=candidate_id,
            output_concat_func=np.concatenate,
    )
    return TopKEncoder(
            model.query_encoder,
            topk_layer=strategy,
            k=k,
            candidates=candidates_embeddings,
            target=model.encoder._schema.select_by_tag(candidate_id).first.name,
    )


# Top-K evaluation


def get_candidates(data, tags=[Tags.ITEM, Tags.ITEM_ID]):
    candidate_features = unique_rows_by_features(data, *tags)
    print(f"Candidate set rows:", candidate_features.num_rows)
    return candidate_features


def recommendation_centric_metrics(topk_model, ds, catalog, batch_size=2048):
    candidate_features = get_candidates(ds, [Tags.USER, Tags.USER_ID])
    catalog_items = set(catalog.keys())

    novelty_scores = []
    unique_recommended_items = set()
    eval_loader = mm.Loader(candidate_features, batch_size=batch_size, shuffle=False)
    _iter = iter(eval_loader)
    for batch, _ in _iter:
        recs = topk_model(batch)[1].numpy()
        # coverage
        # Flatten the list of recommended items and convert it to a set
        unique_recommended_items = unique_recommended_items.union(set(item for u_rec in recs for item in u_rec))
        # novelty
        rec_popularity = [catalog[item[0]] for item in recs if item[0] in catalog_items]
        if rec_popularity:
            novelty_scores.append(1 / np.mean(rec_popularity))
    # Calculate overall novelty
    overall_novelty = np.mean(novelty_scores)

    # Calculate the intersection of unique recommended items and catalog items
    covered_items = unique_recommended_items.intersection(catalog_items)

    # Calculate the catalog coverage
    coverage = len(covered_items) / len(catalog)
    return dict(coverage=coverage, novelty=overall_novelty)


def evaluate_model(model, ds, schema, topk=10, batch_size=1024, item_id="movie_id"):
    candidate_features = get_candidates(ds)

    topk_model = to_top_k_encoder(
            model, candidate_features, k=topk, batch_size=batch_size
    )
    topk_model.compile(run_eagerly=False, metrics=topk_metrics_aggregator())

    eval_loader = mm.Loader(ds, batch_size=batch_size).map(mm.ToTarget(schema, item_id))
    catalog = ds.to_ddf().compute()[item_id].value_counts().to_dict()
    return topk_model.evaluate(
            eval_loader, return_dict=True
    ) | recommendation_centric_metrics(topk_model, ds, catalog)


class EvaluationCallback(Callback):
    def __init__(self, period, topk, data, type_, schema):
        super(EvaluationCallback, self).__init__()
        self.period = period
        self.topk = topk
        self.data = data
        self.type_ = type_
        self.schema = schema
        self.records = defaultdict(list)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:
            # Perform custom actions at the specified frequency
            res = evaluate_model(self.model, self.data, self.schema, topk=self.topk)
            for k, m in res.items():
                k = "auc" if k.startswith("auc") else k
                self.records[k].append(m)
            print(f"{self.type_} dataset topk evaluation {res}")

    def get_records(self):
        return self.records
