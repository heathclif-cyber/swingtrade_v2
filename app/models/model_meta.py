from app.extensions import db, utcnow


class ModelMeta(db.Model):
    __tablename__ = "model_meta"
    __table_args__ = (
        db.UniqueConstraint("coin_id", "model_type", name="uq_model_meta_coin_type"),
    )

    id                    = db.Column(db.Integer, primary_key=True)
    coin_id               = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False)
    model_type            = db.Column(db.Text, nullable=False)   # lgbm, lstm, ensemble
    accuracy              = db.Column(db.Float)
    sharpe_ratio          = db.Column(db.Float)
    profit_factor         = db.Column(db.Float)
    f1_macro              = db.Column(db.Float)
    win_rate              = db.Column(db.Float)
    total_trades          = db.Column(db.Integer)
    max_drawdown          = db.Column(db.Float)
    model_path            = db.Column(db.Text)
    scaler_path           = db.Column(db.Text)
    meta_learner_path     = db.Column(db.Text)
    calibrator_path       = db.Column(db.Text)
    inference_config_path = db.Column(db.Text)
    n_features            = db.Column(db.Integer, default=85)
    status                = db.Column(db.Text, default="available")  # available, active, deprecated
    trained_at            = db.Column(db.DateTime(timezone=True))
    evaluated_at          = db.Column(db.DateTime(timezone=True))

    coin     = db.relationship("Coin", back_populates="model_metas")
    signals  = db.relationship("Signal", back_populates="model_meta", lazy="dynamic")
    selections = db.relationship("ModelSelection", back_populates="model_meta", lazy="dynamic")

    def __repr__(self):
        return f"<ModelMeta {self.model_type} coin_id={self.coin_id}>"
