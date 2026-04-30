from app.extensions import db, utcnow


class Coin(db.Model):
    __tablename__ = "coin"

    id             = db.Column(db.Integer, primary_key=True)
    symbol         = db.Column(db.Text, nullable=False, unique=True)
    status         = db.Column(db.Text, default="active")   # active, inactive, delisted
    last_signal_at = db.Column(db.DateTime(timezone=True))
    last_updated   = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    model_metas    = db.relationship("ModelMeta", back_populates="coin", lazy="dynamic")
    signals        = db.relationship("Signal", back_populates="coin", lazy="dynamic")
    trades         = db.relationship("Trade", back_populates="coin", lazy="dynamic")
    model_selection = db.relationship("ModelSelection", back_populates="coin", uselist=False)
    performance_summaries = db.relationship("PerformanceSummary", back_populates="coin", lazy="dynamic")

    def __repr__(self):
        return f"<Coin {self.symbol}>"
