from datetime import datetime, timezone

from app.extensions import db, utcnow


class PerformanceSummary(db.Model):
    __tablename__ = "performance_summary"

    id           = db.Column(db.Integer, primary_key=True)
    coin_id      = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False)
    period       = db.Column(db.Text, nullable=False)   # 7d, 30d, all
    total_trades = db.Column(db.Integer)
    win_count    = db.Column(db.Integer)
    loss_count   = db.Column(db.Integer)
    win_rate     = db.Column(db.Float)
    total_pnl    = db.Column(db.Float)
    avg_pnl      = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    snapshot_at  = db.Column(db.DateTime(timezone=True), nullable=False,
                             default=lambda: datetime.now(timezone.utc))
    updated_at   = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    coin = db.relationship("Coin", back_populates="performance_summaries")

    def __repr__(self):
        return f"<PerformanceSummary coin_id={self.coin_id} period={self.period} @ {self.snapshot_at}>"
