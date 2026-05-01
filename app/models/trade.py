from app.extensions import db, utcnow


class Trade(db.Model):
    __tablename__ = "trade"

    id          = db.Column(db.Integer, primary_key=True)
    signal_id   = db.Column(db.Integer, db.ForeignKey("signal.id"))
    coin_id     = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False)
    direction   = db.Column(db.Text, nullable=False)        # LONG, SHORT
    entry_price = db.Column(db.Float, nullable=False)
    exit_price  = db.Column(db.Float)
    tp_price      = db.Column(db.Float)
    sl_price      = db.Column(db.Float)
    h4_swing_high = db.Column(db.Float)
    h4_swing_low  = db.Column(db.Float)
    quantity      = db.Column(db.Float, default=1.0)
    leverage    = db.Column(db.Float, default=3.0)
    fee_total   = db.Column(db.Float, default=0.0)
    pnl_gross   = db.Column(db.Float)
    pnl_net     = db.Column(db.Float)
    pnl_pct     = db.Column(db.Float)
    exit_reason = db.Column(db.Text)   # tp_hit, sl_hit, time_exit, manual_close
    status      = db.Column(db.Text, default="open")        # open, closed
    opened_at   = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    closed_at   = db.Column(db.DateTime(timezone=True))
    hold_bars   = db.Column(db.Integer)

    signal = db.relationship("Signal", back_populates="trade")
    coin   = db.relationship("Coin", back_populates="trades")

    __table_args__ = (
        db.Index("idx_trade_status", "status"),
        db.Index("idx_trade_coin", "coin_id", opened_at.desc()),
    )

    def __repr__(self):
        return f"<Trade {self.direction} coin_id={self.coin_id} status={self.status}>"
