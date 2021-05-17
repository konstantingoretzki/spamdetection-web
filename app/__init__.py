from flask import Flask

# General Flask setup
app = Flask(__name__)
app.config["SECRET_KEY"] = "e6slHBXq2hHaLPaHpDUdpaId5b6GyrO9"

from app import routes
