from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class TrafficData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    car_count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())