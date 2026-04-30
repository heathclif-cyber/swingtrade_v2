import os
import sys
from dotenv import load_dotenv

# Pastikan path aplikasi dikenali
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv()

from app import create_app
from app.extensions import db
import app.models  # noqa: F401

def reset_and_seed_database():
    app = create_app()
    with app.app_context():
        print(f"Menggunakan database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        print("\nMenghapus semua tabel (DROP ALL)...")
        db.drop_all()
        print("Berhasil menghapus tabel.")
        
        print("\nMembuat ulang semua tabel (CREATE ALL)...")
        db.create_all()
        print("Berhasil membuat tabel.")
        
        print("\nMenjalankan proses seeding (AUTO-SEED)...")
        from app.models.coin import Coin
        from app.__init__ import _auto_seed
        
        if Coin.query.count() == 0:
            _auto_seed(app)
            print("Seeding selesai!")
        else:
            print("Tabel Coin tidak kosong, skip seeding.")

if __name__ == "__main__":
    reset_and_seed_database()
