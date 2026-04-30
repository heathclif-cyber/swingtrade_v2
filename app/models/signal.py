from app.extensions import db, utcnow


class Signal(db.Model):
    __tablename__ = "signal"

    id               = db.Column(db.Integer, primary_key=True)
    coin_id          = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False)
    model_meta_id    = db.Column(db.Integer, db.ForeignKey("model_meta.id"))
    direction        = db.Column(db.Text, nullable=False)   # LONG, SHORT, FLAT
    confidence       = db.Column(db.Float)                  # 0.0 - 1.0
    entry_price      = db.Column(db.Float)
    tp_price         = db.Column(db.Float)
    sl_price         = db.Column(db.Float)
    atr_at_signal    = db.Column(db.Float)
    timeframe        = db.Column(db.Text, default="1h")
    feature_snapshot = db.Column(db.Text)                   # JSON string
    signal_time      = db.Column(db.DateTime(timezone=True), nullable=False)
    created_at       = db.Column(db.DateTime(timezone=True), default=utcnow)

    coin       = db.relationship("Coin", back_populates="signals")
    model_meta = db.relationship("ModelMeta", back_populates="signals")
    trade      = db.relationship("Trade", back_populates="signal", uselist=False)

    __table_args__ = (
        db.Index("idx_signal_coin_time", "coin_id", signal_time.desc()),
    )

    def __repr__(self):
        return f"<Signal {self.direction} coin_id={self.coin_id} conf={self.confidence}>"
