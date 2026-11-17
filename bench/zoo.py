"""Deterministic model zoo: industry-baseline models per framework.

Baselines follow industry-standard reference workloads:

  TABULAR (sklearn / xgboost / lightgbm)
    - "higgs10"  (10 features)  — the top-10 jet features of the HIGGS
      dataset, the canonical XGBoost/LightGBM tabular benchmark.
    - "higgs"    (28 features)  — full HIGGS schema, the headline
      XGBoost and LightGBM benchmark dataset.
    - "covertype" (54 features, 7 classes) — sklearn's canonical
      large-scale classification benchmark (Forest CoverType).
    Estimators per framework are the canonical first-choice models:
    sklearn LogisticRegression (linear baseline) / RandomForest /
    GradientBoosting; XGBoost XGBClassifier / XGBRegressor (hist);
    LightGBM LGBMClassifier / LGBMRegressor.

  VISION (onnx)
    - "mnist"    (1x28x28) — canonical small vision input.
    - "cifar"    (3x32x32) — canonical medium vision input.
    - "imagenet" (3x224x224) — canonical large vision input
      (ResNet-class serving shape).
    Graphs: dense MLP (tabular) + CNN (vision), ONNX opset 17,
    compatible with CPUExecutionProvider and CUDAExecutionProvider.

  PAYLOADS
    - Validation batch: 8 rows with edge cases (zeros, negatives,
      extreme floats, subnormals) — the parity gate's input.
    - Benchmark payloads: rows per request in {1, 8, 32, 128, 512}
      (the industry serving convention for single-request batch sizes).

Every artifact is generated from fixed seeds: same seed => byte-identical
model + payloads on any machine. SHA-256 manifests gate every run.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20261114
ZOO_ROOT_NAME = "models-zoo"
PAYLOAD_ROWS = [1, 8, 32, 128, 512]

# tier -> (tabular schema, n_train_rows, vision schema)
TIER_BASELINES = {
    "small": {
        "tabular": ("higgs10", 10, 2),  # schema, n_features, n_classes
        "vision": ("mnist", (1, 28, 28), 10),
        "train_rows": 1_000,
    },
    "medium": {
        "tabular": ("higgs", 28, 2),
        "vision": ("cifar", (3, 32, 32), 10),
        "train_rows": 10_000,
    },
    "large": {
        "tabular": ("covertype", 54, 7),
        "vision": ("imagenet", (3, 224, 224), 1000),
        "train_rows": 50_000,
    },
}

IMPLEMENTATIONS = {
    "sklearn": "mlserver_sklearn.SKLearnModel",
    "xgboost": "mlserver_xgboost.XGBoostModel",
    "lightgbm": "mlserver_lightgbm.LightGBMModel",
    "onnx": "mlserver_onnx.OnnxModel",
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _tier_index(tier: str) -> int:
    return ["small", "medium", "large"].index(tier)


def _tabular_dataset(tier: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    schema, n_features, n_classes = TIER_BASELINES[tier]["tabular"]
    rng = _rng(SEED + _tier_index(tier))
    x = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    w = rng.standard_normal((n_features, n_classes)).astype(np.float32)
    y = np.argmax(x @ w, axis=1).astype(np.int64)
    return x, y


# --------------------------------------------------------------------------
# Framework builders — each returns {variant_name: artifact_bytes}
# --------------------------------------------------------------------------


def build_sklearn(tier: str) -> dict[str, Any]:
    import io

    import joblib
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = _tabular_dataset(tier, TIER_BASELINES[tier]["train_rows"])
    depth = {"small": 4, "medium": 8, "large": 12}[tier]
    n_estimators = 20 * (1 + _tier_index(tier))
    models = {
        "pipeline-logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, random_state=SEED)),
            ]
        ),
        "random-forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            random_state=SEED,
            n_jobs=1,
        ),
        "grad-boost": GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=3, random_state=SEED
        ),
    }
    artifacts = {}
    for name, model in models.items():
        target = y if name != "grad-boost" else y.astype(np.float64)
        model.fit(x, target)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        artifacts[name] = buffer.getvalue()
    return artifacts


def build_xgboost(tier: str) -> dict[str, Any]:
    import xgboost as xgb

    x, y = _tabular_dataset(tier, TIER_BASELINES[tier]["train_rows"])
    _, n_features, n_classes = TIER_BASELINES[tier]["tabular"]
    depth = {"small": 3, "medium": 6, "large": 9}[tier]
    trees = {"small": 20, "medium": 60, "large": 150}[tier]
    params = {
        "max_depth": depth,
        "nthread": 1,
        "seed": SEED,
        "tree_method": "hist",
    }
    artifacts = {}
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": n_classes, **params},
        xgb.DMatrix(x, label=y),
        num_boost_round=trees,
    )
    artifacts["classifier"] = booster.save_raw("json")
    booster_r = xgb.train(
        {"objective": "reg:squarederror", **params},
        xgb.DMatrix(x, label=x[:, 0] * 2.0 + 1.0),
        num_boost_round=trees,
    )
    artifacts["regressor"] = booster_r.save_raw("json")
    return artifacts


def build_lightgbm(tier: str) -> dict[str, Any]:
    import lightgbm as lgb

    x, y = _tabular_dataset(tier, TIER_BASELINES[tier]["train_rows"])
    _, n_features, n_classes = TIER_BASELINES[tier]["tabular"]
    depth = {"small": 4, "medium": 8, "large": 12}[tier]
    trees = {"small": 20, "medium": 60, "large": 150}[tier]
    base = {
        "max_depth": depth,
        "seed": SEED,
        "num_threads": 1,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    artifacts = {}
    booster = lgb.train(
        {"objective": "multiclass", "num_class": n_classes, **base},
        lgb.Dataset(x, label=y),
        num_boost_round=trees,
    )
    artifacts["classifier"] = booster.model_to_string().encode()
    booster_r = lgb.train(
        {"objective": "regression", **base},
        lgb.Dataset(x, label=x[:, 0] * 2.0 + 1.0),
        num_boost_round=trees,
    )
    artifacts["regressor"] = booster_r.model_to_string().encode()
    return artifacts


def _onnx_model(nodes, initializers, input_spec, output_spec) -> bytes:
    from onnx import helper

    graph = helper.make_graph(nodes, "net", input_spec, output_spec, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


def build_onnx_mlp(tier: str) -> bytes:
    """Dense MLP on the tier's tabular schema (CPU + CUDA compatible)."""
    from onnx import TensorProto, helper

    _, n_features, n_classes = TIER_BASELINES[tier]["tabular"]
    hidden = {"small": 32, "medium": 128, "large": 512}[tier]
    rng = _rng(SEED + 7)
    initializers, nodes = [], []
    dims = [n_features, hidden, hidden // 2, n_classes]
    for index in range(len(dims) - 1):
        weight = (rng.standard_normal((dims[index], dims[index + 1])) * 0.05).astype(
            np.float32
        )
        bias = np.zeros(dims[index + 1], dtype=np.float32)
        initializers += [
            helper.make_tensor(
                f"w{index}", TensorProto.FLOAT, weight.shape, weight.flatten()
            ),
            helper.make_tensor(
                f"b{index}", TensorProto.FLOAT, bias.shape, bias.flatten()
            ),
        ]
        source = "input" if index == 0 else f"a{index - 1}"
        last = index == len(dims) - 2
        nodes.append(
            helper.make_node(
                "Gemm",
                [source, f"w{index}", f"b{index}"],
                ["output"] if last else [f"h{index}"],
                name=f"gemm{index}",
            )
        )
        if index < len(dims) - 2:
            nodes.append(
                helper.make_node(
                    "Relu", [f"h{index}"], [f"a{index}"], name=f"relu{index}"
                )
            )
    return _onnx_model(
        nodes,
        initializers,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, n_classes])],
    )


def build_onnx_cnn(tier: str) -> bytes:
    """CNN at the tier's canonical vision shape (CUDA kernels on GPU)."""
    from onnx import TensorProto, helper

    schema, shape, n_classes = TIER_BASELINES[tier]["vision"]
    channels = {"small": 8, "medium": 16, "large": 32}[tier]
    # conv 3x3 (no padding): size -> size - 2; then maxpool 2x2
    pooled = (shape[1] - 2) // 2 * (shape[2] - 2) // 2
    rng = _rng(SEED + 11)
    conv_w = (rng.standard_normal((channels, shape[0], 3, 3)) * 0.05).astype(np.float32)
    fc_w = (
        rng.standard_normal((channels * pooled, min(n_classes, 100))) * 0.05
    ).astype(np.float32)
    out_classes = min(n_classes, 100)
    initializers = [
        helper.make_tensor("conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.flatten()),
        helper.make_tensor("fc_w", TensorProto.FLOAT, fc_w.shape, fc_w.flatten()),
        helper.make_tensor(
            "shape",
            TensorProto.INT64,
            [2],
            np.array([-1, channels * pooled], dtype=np.int64),
        ),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "conv_w"], ["c0"], name="conv"),
        helper.make_node("Relu", ["c0"], ["r0"], name="relu"),
        helper.make_node(
            "MaxPool", ["r0"], ["p0"], kernel_shape=[2, 2], strides=[2, 2], name="pool"
        ),
        helper.make_node("Reshape", ["p0", "shape"], ["flat"], name="flatten"),
        helper.make_node("MatMul", ["flat", "fc_w"], ["output"], name="fc"),
    ]
    return _onnx_model(
        nodes,
        initializers,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, *shape])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [None, out_classes]
            )
        ],
    )


def build_onnx_nan(tier: str) -> bytes:
    """NaN-emitting tabular model: Sub(input, input) -> zeros, then
    Div(zeros, zeros) -> IEEE NaN everywhere.

    Purpose: the codec suite's NaN axis. MLServer's numpy codec
    converts NaN outputs to JSON null in a PER-ELEMENT Python loop
    (codecs/numpy.py) — this model maximises that path so its cost is
    measurable against the clean-float baseline (mlp).
    """
    from onnx import TensorProto, helper

    n_features = TIER_BASELINES[tier]["tabular"][1]
    nodes = [
        helper.make_node("Sub", ["input", "input"], ["zeros"], name="make_zeros"),
        helper.make_node("Div", ["zeros", "zeros"], ["output"], name="make_nan"),
    ]
    return _onnx_model(
        nodes,
        [],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [None, n_features]
            )
        ],
    )


def build_onnx(tier: str) -> dict[str, Any]:
    return {
        "mlp": build_onnx_mlp(tier),
        "cnn": build_onnx_cnn(tier),
        "nan": build_onnx_nan(tier),
    }


BUILDERS = {
    "sklearn": build_sklearn,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "onnx": build_onnx,
}

ARTIFACT_NAMES = {
    ("sklearn", "pipeline-logreg"): "model.joblib",
    ("sklearn", "random-forest"): "model.joblib",
    ("sklearn", "grad-boost"): "model.joblib",
    ("xgboost", "classifier"): "model.bst",
    ("xgboost", "regressor"): "model.bst",
    ("lightgbm", "classifier"): "model.bst",
    ("lightgbm", "regressor"): "model.bst",
    ("onnx", "mlp"): "model.onnx",
    ("onnx", "cnn"): "model.onnx",
    ("onnx", "nan"): "model.onnx",
}


# --------------------------------------------------------------------------
# Settings, validation batches, benchmark payloads
# --------------------------------------------------------------------------


def _input_spec(framework: str, tier: str, variant: str) -> list[dict]:
    if framework == "onnx" and variant == "cnn":
        shape = list(TIER_BASELINES[tier]["vision"][1])
        return [{"name": "input", "datatype": "FP32", "shape": [-1, *shape]}]
    n_features = TIER_BASELINES[tier]["tabular"][1]
    return [{"name": "input", "datatype": "FP32", "shape": [-1, n_features]}]


CLASSIFIER_VARIANTS = {"pipeline-logreg", "random-forest", "classifier"}
REGRESSOR_VARIANTS = {"grad-boost", "regressor"}


def _output_spec(framework: str, tier: str, variant: str) -> list[dict]:
    if framework == "onnx":
        if variant == "cnn":
            n_classes = min(TIER_BASELINES[tier]["vision"][2], 100)
        elif variant == "nan":
            # NaN model emits one output per input feature (all NaN)
            n_classes = TIER_BASELINES[tier]["tabular"][1]
        else:
            n_classes = TIER_BASELINES[tier]["tabular"][2]
        return [{"name": "output", "datatype": "FP32", "shape": [-1, n_classes]}]
    # sklearn / xgboost / lightgbm: output names are runtime methods;
    # requests without an outputs section default to "predict".
    if framework == "sklearn" and variant in CLASSIFIER_VARIANTS:
        # sklearn predict -> integer class labels, 1-D
        return [{"name": "predict", "datatype": "INT64", "shape": [-1]}]
    if variant in REGRESSOR_VARIANTS:
        # regressor predict -> 1-D floats
        return [{"name": "predict", "datatype": "FP32", "shape": [-1]}]
    # xgboost / lightgbm classifiers: predict -> class probabilities
    n_classes = TIER_BASELINES[tier]["tabular"][2]
    return [{"name": "predict", "datatype": "FP32", "shape": [-1, n_classes]}]


def _model_settings(framework: str, tier: str, variant: str) -> dict[str, Any]:
    artifact = ARTIFACT_NAMES[(framework, variant)]
    return {
        "name": f"{framework}-{variant}-{tier}",
        "implementation": IMPLEMENTATIONS[framework],
        "parameters": {"uri": artifact},
        "inputs": _input_spec(framework, tier, variant),
        "outputs": _output_spec(framework, tier, variant),
    }


def _validation_batch(framework: str, tier: str, variant: str) -> dict[str, Any]:
    """Deterministic parity batch WITH edge cases (zeros, negatives,
    extreme floats, subnormals)."""
    if framework == "onnx" and variant == "cnn":
        shape = TIER_BASELINES[tier]["vision"][1]
        rng = _rng(SEED + 23)
        rows = rng.standard_normal((8, *shape)).astype(np.float32)
        rows[0] = 0.0
        rows[1] = -np.abs(rows[1])
        rows[2] *= 1e3
        return {"input": rows}
    n_features = TIER_BASELINES[tier]["tabular"][1]
    rng = _rng(SEED + 23)
    rows = rng.standard_normal((8, n_features)).astype(np.float32)
    rows[0] = 0.0
    rows[1] = -np.abs(rows[1])
    rows[2] *= 1e3
    rows[3] = np.nextafter(np.float32(0), np.float32(1))  # smallest subnormal
    return {"input": rows}


def _benchmark_payloads(framework: str, tier: str, variant: str) -> dict[int, Any]:
    """One payload per industry-standard rows-per-request size."""
    if framework == "onnx" and variant == "cnn":
        shape = TIER_BASELINES[tier]["vision"][1]
        rng = _rng(SEED + 31)
        return {
            rows: {"input": rng.standard_normal((rows, *shape)).astype(np.float32)}
            for rows in PAYLOAD_ROWS
        }
    n_features = TIER_BASELINES[tier]["tabular"][1]
    rng = _rng(SEED + 31)
    return {
        rows: {"input": rng.standard_normal((rows, n_features)).astype(np.float32)}
        for rows in PAYLOAD_ROWS
    }


# --------------------------------------------------------------------------
# Materialization: on-demand generate/download, reuse-if-present
#
# Definitive file layout (gitignored — everything below is reproducible
# from seeds or the pinned download registry):
#
#   models-zoo/{framework}-{variant}-{tier}/
#       model.joblib | model.bst | model.txt | model.onnx   # the model
#       model-settings.json      # MLServer serving settings
#       validation.npz           # parity batch (edge cases)
#       payloads.npz             # benchmark payloads (rows{N} arrays)
#       MANIFEST.json            # per-model SHA-256 contract
# --------------------------------------------------------------------------

DOWNLOAD_REGISTRY: dict[str, dict[str, Any]] = {
    "onnx-mnist-real": {
        "framework": "onnx",
        "tier": "small",
        "variant": "mnist-real",
        "artifact": "model.onnx",
        "url": (
            "https://media.githubusercontent.com/media/onnx/models/main/"
            "validated/vision/classification/mnist/model/mnist-8.onnx"
        ),
        "sha256": ("2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"),
        "settings": {
            "name": "onnx-mnist-real-small",
            "implementation": "mlserver_onnx.OnnxModel",
            "uri": "model.onnx",
            "inputs": [{"name": "Input3", "datatype": "FP32", "shape": [1, 1, 28, 28]}],
            "outputs": [
                {"name": "Plus214_Output_0", "datatype": "FP32", "shape": [1, 10]}
            ],
        },
        "payload_rows": [1],  # static batch-1 graph
        "source_note": "ONNX Model Zoo mnist-8 (opset 8, validated)",
    }
}


def model_specs(
    frameworks: list[str], tiers: list[str], include_real: bool = False
) -> list[dict[str, Any]]:
    """Declarative catalog of every zoo model in scope."""
    specs: list[dict[str, Any]] = []
    for framework in frameworks:
        for tier in tiers:
            for variant, artifact in BUILDERS[framework](tier).items():
                specs.append(
                    {
                        "key": f"{framework}-{variant}-{tier}",
                        "framework": framework,
                        "tier": tier,
                        "variant": variant,
                        "source": "generate",
                        "artifact": ARTIFACT_NAMES[(framework, variant)],
                        "artifact_bytes": artifact,
                        "settings": _model_settings(framework, tier, variant),
                        "validation": _validation_batch(framework, tier, variant),
                        "payloads": _benchmark_payloads(framework, tier, variant),
                    }
                )
    if include_real:
        for key, entry in DOWNLOAD_REGISTRY.items():
            if entry["framework"] in frameworks and entry["tier"] in tiers:
                specs.append(
                    {
                        "key": f"{key}-{entry['tier']}",
                        "framework": entry["framework"],
                        "tier": entry["tier"],
                        "variant": entry["variant"],
                        "source": "download",
                        "registry_key": key,
                        "artifact": entry["artifact"],
                        "settings": entry["settings"],
                        "validation": _real_validation_batch(entry),
                        "payloads": _real_payloads(entry),
                    }
                )
    return specs


def _real_validation_batch(entry: dict[str, Any]) -> dict[str, Any]:
    name = entry["settings"]["inputs"][0]["name"]
    shape = entry["settings"]["inputs"][0]["shape"]
    rng = _rng(SEED + 53)
    if shape[0] == -1:  # dynamic batch: 8 edge-case rows
        rows = rng.standard_normal((8, *shape[1:])).astype(np.float32)
        rows[0] = 0.0
        rows[1] = -np.abs(rows[1])
        rows[2] *= 1e3
        return {name: rows}
    # static batch: exactly the declared shape, with edge-case values
    tensor = rng.standard_normal(shape).astype(np.float32)
    flat = tensor.reshape(-1)
    if flat.size >= 4:
        flat[0] = 0.0
        flat[1] = -abs(flat[1])
        flat[2] *= 1e3
        flat[3] = np.nextafter(np.float32(0), np.float32(1))
    return {name: tensor.reshape(shape)}


def _real_payloads(entry: dict[str, Any]) -> dict[int, Any]:
    name = entry["settings"]["inputs"][0]["name"]
    shape = entry["settings"]["inputs"][0]["shape"]
    rng = _rng(SEED + 59)
    if shape[0] == -1:  # dynamic batch
        return {
            rows: {name: rng.standard_normal((rows, *shape[1:])).astype(np.float32)}
            for rows in entry.get("payload_rows", PAYLOAD_ROWS)
        }
    # static batch: only payloads matching the declared batch size
    static_batch = shape[0]
    return {
        rows: {name: rng.standard_normal((rows, *shape[1:])).astype(np.float32)}
        for rows in entry.get("payload_rows", PAYLOAD_ROWS)
        if rows == static_batch
    }


def _download(url: str, expected_sha256: str, destination: Path) -> None:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=300) as response:
        temporary.write_bytes(response.read())
    actual = _sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded model failed SHA-256 verification: "
            f"expected {expected_sha256[:16]}... got {actual[:16]}... "
            f"({url})"
        )
    import os

    os.replace(temporary, destination)


def _write_model(model_dir: Path, spec: dict[str, Any]) -> None:
    """Write one model directory: artifact + settings + batches + manifest."""
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / spec["artifact"]
    if spec["source"] == "download":
        entry = DOWNLOAD_REGISTRY[spec["registry_key"]]
        _download(entry["url"], entry["sha256"], artifact_path)
    else:
        artifact_path.write_bytes(spec["artifact_bytes"])
    (model_dir / "model-settings.json").write_text(
        json.dumps(spec["settings"], indent=2) + "\n"
    )
    np.savez(model_dir / "validation.npz", **spec["validation"])
    np.savez(
        model_dir / "payloads.npz",
        **{f"rows{rows}": array for rows, array in spec["payloads"].items()},
    )
    manifest = {
        "model": spec["key"],
        "source": spec["source"],
        "files": {
            path.name: _sha256(path)
            for path in sorted(model_dir.iterdir())
            if path.name != "MANIFEST.json"
        },
    }
    if spec["source"] == "download":
        manifest["registry"] = {
            "url": DOWNLOAD_REGISTRY[spec["registry_key"]]["url"],
            "sha256": DOWNLOAD_REGISTRY[spec["registry_key"]]["sha256"],
        }
    else:
        manifest["seed"] = SEED
    (model_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _model_ok(model_dir: Path) -> bool:
    """A model directory is reusable when its manifest verifies."""
    manifest_path = model_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    for file_name, expected in manifest.get("files", {}).items():
        path = model_dir / file_name
        if not path.is_file() or _sha256(path) != expected:
            return False
    return True


ZOO_README = """# Model Zoo (auto-materialized — gitignored)

Everything here is reproducible: synthetic models from fixed seeds, real
reference models from the pinned download registry. It is materialized
on demand by the benchmark runner; never edit by hand.

Definitive layout per model:

    models-zoo/{{framework}}-{{variant}}-{{tier}}/
        model.joblib | model.bst | model.txt | model.onnx   # the model
        model-settings.json      # MLServer serving settings
        validation.npz           # parity batch (edge cases)
        payloads.npz             # benchmark payloads (rows{{N}} arrays)
        MANIFEST.json            # per-model SHA-256 contract

Baselines: {baselines}

To force regeneration: rm -rf models-zoo (everything rebuilds
deterministically), or make models-clean.
"""


def zoo_root(repo_root: Path) -> Path:
    return repo_root / ZOO_ROOT_NAME


def materialize(
    repo_root: Path, frameworks: list[str], tiers: list[str], include_real: bool = False
) -> Path:
    """Ensure every in-scope model exists and verifies; generate or
    download only what is missing (reuse-if-present)."""
    root = zoo_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    reused, built, downloaded = 0, 0, 0
    for spec in model_specs(frameworks, tiers, include_real):
        model_dir = root / spec["key"]
        if _model_ok(model_dir):
            reused += 1
            continue
        # Stale/partial directories are rebuilt from scratch.
        if model_dir.exists():
            import shutil

            shutil.rmtree(model_dir)
        _write_model(model_dir, spec)
        if spec["source"] == "download":
            downloaded += 1
        else:
            built += 1
    print(f"[zoo] reused={reused} generated={built} downloaded={downloaded}")
    write_manifest(root, frameworks, tiers)
    (root / "README.md").write_text(
        ZOO_README.format(
            baselines=json.dumps(
                {tier: TIER_BASELINES[tier]["tabular"][0] for tier in tiers}
            )
        )
    )
    return root


def clean(repo_root: Path) -> None:
    import shutil

    root = zoo_root(repo_root)
    if root.exists():
        shutil.rmtree(root)
        print(f"[zoo] removed {root}")


# --------------------------------------------------------------------------
# Legacy entry points (kept for compatibility)
# --------------------------------------------------------------------------


def generate(repo_root: Path, frameworks: list[str], tiers: list[str]) -> Path:
    """Generate (idempotently) the zoo; returns the zoo root."""
    return materialize(repo_root, frameworks, tiers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, frameworks: list[str], tiers: list[str]) -> dict:
    manifest: dict[str, Any] = {
        "seed": SEED,
        "payload_rows": PAYLOAD_ROWS,
        "baselines": {
            tier: {
                "tabular": TIER_BASELINES[tier]["tabular"],
                "vision": TIER_BASELINES[tier]["vision"][:2],
            }
            for tier in tiers
        },
        "models": {},
    }
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        manifest["models"][model_dir.name] = {
            path.name: _sha256(path)
            for path in sorted(model_dir.iterdir())
            if path.name != "MANIFEST.json"
        }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(root: Path) -> list[str]:
    """Verify every model's SHA-256 against the manifest."""
    problems: list[str] = []
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        return [f"zoo manifest missing: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    for model_name, files in manifest["models"].items():
        for file_name, expected in files.items():
            path = root / model_name / file_name
            if not path.is_file():
                problems.append(f"{model_name}/{file_name}: missing")
            elif _sha256(path) != expected:
                problems.append(f"{model_name}/{file_name}: SHA-256 mismatch")
    return problems


def list_models(root: Path, framework: str, tier: str) -> list[Path]:
    return sorted(root.glob(f"{framework}-*-{tier}"))
