"""HTTP server for exposing Prometheus metrics.

Provides a lightweight HTTP server to expose RFSN metrics for Prometheus scraping.

Example:
    from rfsn_controller.metrics_server import start_metrics_server
    
    # Start metrics server on port 9090
    start_metrics_server(port=9090)
    
    # Or run in background thread
    import threading
    thread = threading.Thread(target=start_metrics_server, args=(9090,), daemon=True)
    thread.start()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import REGISTRY, generate_latest

logger = logging.getLogger(__name__)

__all__ = ["MetricsHandler", "start_metrics_server", "create_metrics_app"]


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving Prometheus metrics.
    
    Responds to GET /metrics with Prometheus-formatted metrics.
    """
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/metrics":
            # Generate metrics output
            try:
                metrics = generate_latest(REGISTRY)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.end_headers()
                self.wfile.write(metrics)
            except Exception as e:
                logger.error(f"Error generating metrics: {e}")
                self.send_error(500, f"Internal Server Error: {e}")
        
        elif self.path == "/health" or self.path == "/":
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
        
        else:
            self.send_error(404, "Not Found")
    
    def log_message(self, format: str, *args) -> None:
        """Override to use Python logging instead of print."""
        logger.info(f"{self.address_string()} - {format % args}")


def start_metrics_server(port: int = 9090, host: str = "0.0.0.0") -> None:
    """Start HTTP server for Prometheus metrics.
    
    This function blocks until the server is stopped (Ctrl+C).
    
    Args:
        port: Port to listen on (default: 9090)
        host: Host to bind to (default: 0.0.0.0 - all interfaces)
        
    Example:
        from rfsn_controller.metrics_server import start_metrics_server
        
        # Start server (blocking)
        start_metrics_server(port=9090)
        
        # Or run in background
        import threading
        thread = threading.Thread(
            target=start_metrics_server,
            args=(9090,),
            daemon=True
        )
        thread.start()
    """
    server = HTTPServer((host, port), MetricsHandler)
    logger.info(f"Starting metrics server on http://{host}:{port}/metrics")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down metrics server")
        server.shutdown()


def create_metrics_app() -> Callable:
    """Create a WSGI/ASGI compatible metrics application.
    
    Useful for integrating metrics into existing web applications.
    
    Returns:
        WSGI application callable
        
    Example:
        from rfsn_controller.metrics_server import create_metrics_app
        
        # Use with Flask
        from flask import Flask, Response
        app = Flask(__name__)
        
        metrics_app = create_metrics_app()
        
        @app.route('/metrics')
        def metrics():
            return Response(metrics_app(), mimetype='text/plain')
        
        # Use with FastAPI
        from fastapi import FastAPI, Response
        app = FastAPI()
        
        metrics_app = create_metrics_app()
        
        @app.get('/metrics')
        def metrics():
            return Response(
                content=metrics_app(),
                media_type='text/plain'
            )
    """
    def app():
        """Generate metrics output."""
        return generate_latest(REGISTRY)
    
    return app


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for running metrics server standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RFSN Metrics Server - Expose Prometheus metrics"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9090,
        help="Port to listen on (default: 9090)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Start server
    start_metrics_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
