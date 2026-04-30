from app.extensions import db, utcnow


class ModelSelection(db.Model):
    __tablename__ = "model_selection"

    id            = db.Column(db.Integer, primary_key=True)
    coin_id       = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False, unique=True)
    model_meta_id = db.Column(db.Integer, db.ForeignKey("model_meta.id"), nullable=False)
    model_type    = db.Column(db.Text, default="lstm")  # lstm, lgbm, ensemble — default lstm
    selected_at   = db.Column(db.DateTime(timezone=True), default=utcnow)

    coin       = db.relationship("Coin", back_populates="model_selection")
    model_meta = db.relationship("ModelMeta", back_populates="selections")

    def __repr__(self):
        return f"<ModelSelection coin_id={self.coin_id} model_type={self.model_type}>"
