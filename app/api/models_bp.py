from flask import Blueprint, render_template, jsonify, request

bp = Blueprint("models_bp", __name__)


@bp.get("/models")
def models():
    from app.extensions import db
    from app.models.coin import Coin
    from app.models.model_meta import ModelMeta
    from app.models.model_selection import ModelSelection
    from app.services.model_registry import load_registry

    registry_versions = load_registry()

    rows = (
        db.session.query(Coin, ModelSelection, ModelMeta)
        .join(ModelSelection, ModelSelection.coin_id == Coin.id, isouter=True)
        .join(ModelMeta, ModelMeta.id == ModelSelection.model_meta_id, isouter=True)
        .filter(Coin.status == "active")
        .order_by(Coin.symbol)
        .all()
    )

    available_metas = (
        ModelMeta.query.filter_by(status="available")
        .order_by(ModelMeta.run_id.desc())
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
    from config import FEATURE_COLS_V3

    data       = request.get_json(force=True)
    symbol     = data.get("symbol")
    model_type = data.get("model_type", "ensemble")
    run_id     = data.get("run_id")

    if not symbol or not run_id:
        return jsonify({"error": "symbol dan run_id wajib diisi"}), 400

    coin = Coin.query.filter_by(symbol=symbol).first()
    if not coin:
        return jsonify({"error": f"Coin {symbol} tidak ditemukan"}), 404

    meta = ModelMeta.query.filter_by(
        coin_id=coin.id, model_type=model_type, run_id=run_id
    ).first()
    if not meta:
        return jsonify({"error": f"ModelMeta tidak ditemukan untuk {symbol}/{model_type}/{run_id}"}), 404

    # Guard n_features
    if meta.n_features != len(FEATURE_COLS_V3):
        return jsonify({
            "error": f"n_features mismatch: model={meta.n_features}, current={len(FEATURE_COLS_V3)}"
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
    InferenceService.clear_cache(run_id)

    return jsonify({
        "status":     "ok",
        "symbol":     symbol,
        "model_type": model_type,
        "run_id":     run_id,
    })


@bp.get("/api/models/available")
def available_models():
    from app.services.model_registry import load_registry
    return jsonify(load_registry())
