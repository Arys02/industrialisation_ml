from flask import Flask, request, jsonify
import pandas as pd
from pathlib import Path
import os
from src.config import RAW_DATA_DIR, PORT
from src.database_csv import DatabaseCSV

PATH_CSV = RAW_DATA_DIR


def create_app(config=None):
    config = config or {}
    app = Flask(__name__)

    if "CSV_PATH" not in config:
        config["CSV_PATH"] = PATH_CSV

    app.config.update(config)
    db_csv = DatabaseCSV(app.config['CSV_PATH'])

    @app.route('/post_data', methods=['POST'])
    def post_data():
        data = request.json
        db_csv.post_sale(data)

        return jsonify({"status": "success"}), 200

    @app.route('/hello')
    def hello():
        print("youhou")
        return "hello"

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=PORT)
