from flask import Blueprint, render_template, jsonify, request

bp = Blueprint("models_bp", __name__)


def _ensure_cascade_meta(coin, ModelMeta, utcnow, db):
    """Buat ModelMeta cascade dari ensemble (preferred) atau lstm untuk coin ini."""
    source = (
        ModelMeta.query.filter_by(coin_id=coin.id, model_type="ensemble").first()
        or ModelMeta.query.filter_by(coin_id=coin.id, model_type="lstm").first()
    )
    if not source:
        return None

    meta = ModelMeta(
        coin_id            = coin.id,
        model_type         = "cascade",
        accuracy           = source.accuracy,
        sharpe_ratio       = source.sharpe_ratio,
        profit_factor      = source.profit_factor,
        f1_macro           = source.f1_macro,
        win_rate           = source.win_rate,
        total_trades       = source.total_trades,
        max_drawdown       = source.max_drawdown,
        model_path         = source.model_path,
        scaler_path        = source.scaler_path,
        n_features         = source.n_features,
        status             = "available",
        trained_at         = source.trained_at,
        evaluated_at       = source.evaluated_at,
    )
    db.session.add(meta)
    db.session.flush()
    return meta


@bp.get("/models")
def models():
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.models.performance_summary import PerformanceSummary
    from app.services.model_registry import load_registry, load_inference_config
    from sqlalchemy import and_

    registry_versions = load_registry()

    # Rec #2: inject fallback backtest_summary untuk lgbm/lstm dari rata-rata backtest_per_coin
    cfg = load_inference_config()
    bpc = cfg.get("backtest_per_coin", {})
    if bpc:
        vals = list(bpc.values())
        mean_wr = sum(v["winrate"] for v in vals) / len(vals)
        mean_dd = sum(v.get("dd_lev5x", v.get("dd_lev3x", 0.0)) for v in vals) / len(vals)
        for v in registry_versions:
            if not v.get("backtest_summary"):
                v["backtest_summary"] = {
                    "mean_winrate":        round(mean_wr, 4),
                    "mean_drawdown_lev5x": round(mean_dd, 4),
                    "is_avg_fallback":     True,
                }

    rows = (
        db.session.query(Coin, ModelSelection, ModelMeta, PerformanceSummary)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id, isouter=True)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id, isouter=True)
        .join(
            PerformanceSummary,
            and_(PerformanceSummary.coin_id == Coin.id, PerformanceSummary.period == "all"),
            isouter=True,
        )
        .filter(Coin.status == "active")
        .order_by(Coin.symbol)
        .all()
    )

    available_metas = (
        ModelMeta.query.filter_by(status="available")
        .order_by(ModelMeta.model_type)
        .all()
    )

    available_types = sorted(set(m.model_type for m in available_metas))

    # Cascade tersedia jika ada ensemble atau lstm yang bisa jadi dasar
    if ("cascade" not in available_types
            and any(m.model_type in ("ensemble", "lstm") for m in available_metas)):
        available_types.append("cascade")

    return render_template(
        "models.html",
        rows              = rows,
        registry_versions = registry_versions,
        available_metas   = available_metas,
        available_types   = available_types,
    )


@bp.post("/models/select")
def select_model():
    from app.extensions import db, utcnow
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.services.inference import InferenceService
    from app.services.config_loader import get_n_features

    data       = request.get_json(force=True)
    symbol     = data.get("symbol")
    model_type = data.get("model_type", "ensemble")

    if not symbol:
        return jsonify({"error": "symbol wajib diisi"}), 400

    coin = Coin.query.filter_by(symbol=symbol).first()
    if not coin:
        return jsonify({"error": f"Coin {symbol} tidak ditemukan"}), 404

    meta = ModelMeta.query.filter_by(
        coin_id=coin.id, model_type=model_type
    ).first()

    # Cascade: auto-create dari ensemble/lstm jika belum ada
    if not meta and model_type == "cascade":
        meta = _ensure_cascade_meta(coin, ModelMeta, utcnow, db)

    if not meta:
        return jsonify({"error": f"ModelMeta tidak ditemukan untuk {symbol}/{model_type}"}), 404

    # Guard n_features
    if meta.n_features != get_n_features():
        return jsonify({
            "error": f"n_features mismatch: model={meta.n_features}, current={get_n_features()}"
        }), 400

    # Update ModelSelection
    sel = ModelSelection.query.filter_by(coin_id=coin.id).first()
    if sel:
        sel.model_meta_id = meta.id
        sel.selected_at   = utcnow()
    else:
        sel = ModelSelection(coin_id=coin.id, model_meta_id=meta.id)
        db.session.add(sel)

    db.session.commit()

    # Clear cache model lama
    InferenceService.clear_cache()

    return jsonify({
        "status":     "ok",
        "symbol":     symbol,
        "model_type": model_type,
    })


@bp.post("/models/select-all")
def select_all_models():
    """
    Bulk update: ganti model SEMUA coin aktif ke model_type yang sama.
    Body JSON: {"model_type": "lstm"}
    """
    from app.extensions import db, utcnow
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.services.inference import InferenceService
    from app.services.config_loader import get_n_features

    data       = request.get_json(force=True)
    model_type = data.get("model_type", "lstm")

    nf = get_n_features()
    coins = Coin.query.filter_by(status="active").all()
    updated = 0
    errors  = []

    for coin in coins:
        meta = ModelMeta.query.filter_by(
            coin_id=coin.id, model_type=model_type
        ).first()

        # Cascade: auto-create dari ensemble/lstm jika belum ada
        if not meta and model_type == "cascade":
            meta = _ensure_cascade_meta(coin, ModelMeta, utcnow, db)

        if not meta:
            errors.append(f"{coin.symbol}: ModelMeta {model_type} tidak ditemukan")
            continue
        if meta.n_features != nf:
            errors.append(f"{coin.symbol}: n_features mismatch ({meta.n_features} vs {nf})")
            continue

        sel = ModelSelection.query.filter_by(coin_id=coin.id).first()
        if sel:
            sel.model_meta_id = meta.id
            sel.selected_at   = utcnow()
        else:
            sel = ModelSelection(coin_id=coin.id, model_meta_id=meta.id)
            db.session.add(sel)
        updated += 1

    db.session.commit()
    InferenceService.clear_cache()

    return jsonify({
        "status":     "ok",
        "updated":    updated,
        "total":      len(coins),
        "model_type": model_type,
        "errors":     errors,
    })


@bp.get("/api/cascade-config")
def cascade_config_get():
    from app.services.model_registry import load_inference_config
    cfg = load_inference_config()
    return jsonify(cfg.get("cascade", {}).copy())


@bp.post("/api/cascade-config")
def cascade_config_post():
    import json
    from app.services.model_registry import load_inference_config
    from app.services.inference import InferenceService

    body = request.get_json(silent=True) or {}
    errors = []
    validated = {}
    for key in ("scout_flat_threshold", "scout_signal_threshold", "confirmer_threshold"):
        val = body.get(key)
        if val is None:
            errors.append(f"{key} wajib")
        elif not (0.0 <= float(val) <= 1.0):
            errors.append(f"{key} harus antara 0.0 — 1.0")
        else:
            validated[key] = round(float(val), 2)

    if errors:
        return jsonify({"status": "error", "error": "; ".join(errors)}), 422

    # Tulis ke inference_config.json
    from pathlib import Path
    cfg = load_inference_config()
    models_dir = Path(__file__).parent.parent.parent / "models"
    config_path = models_dir / "inference_config.json"
    cfg["cascade"] = validated
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    InferenceService.clear_cache()
    return jsonify({"status": "ok", "cascade": validated})


@bp.get("/api/models/available")
def available_models():
    from app.services.model_registry import load_registry
    return jsonify(load_registry())
