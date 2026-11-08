"""
routes/__init__.py
==================
Exports modular Flask Blueprints for CyberGraph NIDS.
"""

from routes.auth_routes import auth_bp
from routes.admin_routes import admin_bp
from routes.device_routes import device_bp
from routes.stream_routes import stream_bp
from routes.report_routes import report_bp

__all__ = ["auth_bp", "admin_bp", "device_bp", "stream_bp", "report_bp"]
