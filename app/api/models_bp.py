from flask import Blueprint, render_template, jsonify, request

bp = Blueprint("models_bp", __name__)


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
        mean_dd = sum(v["dd_lev3x"] for v in vals) / len(vals)
        for v in registry_versions:
            if not v.get("backtest_summary"):
                v["backtest_summary"] = {
                    "mean_winrate":        round(mean_wr, 4),
                    "mean_drawdown_lev3x": round(mean_dd, 4),
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

    return render_template(
        "models.html",
        rows              = rows,
        registry_versions = registry_versions,
        available_metas   = available_metas,
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


@bp.get("/api/models/available")
def available_models():
    from app.services.model_registry import load_registry
    return jsonify(load_registry())
