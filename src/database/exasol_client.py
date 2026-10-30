"""Exasol connector built on the official `pyexasol` driver. Credentials come ONLY from environment variables.
NOTE: written from pyexasol's documented API; it has NOT been run against a live Exasol instance by the author
(no network access during development). Use scripts/check_exasol.py to verify on your machine."""
import os
import ssl

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sql")


class ExasolUnavailable(Exception):
    pass


def config_from_env() -> dict:
    return dict(dsn=os.getenv("EXASOL_DSN", ""), user=os.getenv("EXASOL_USER", ""), password=os.getenv("EXASOL_PASSWORD", ""),
                schema=os.getenv("EXASOL_SCHEMA", "STOCKSAGE"),
                validate_cert=os.getenv("EXASOL_VALIDATE_CERT", "false").lower() == "true")


def is_configured() -> bool:
    c = config_from_env()
    return bool(c["dsn"] and c["user"] and c["password"])


def _read(name: str) -> str:
    with open(os.path.join(SQL_DIR, name), encoding="utf-8") as f:
        return f.read()


class ExasolClient:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or config_from_env()
        self.conn = None

    def connect(self):
        try:
            import pyexasol
        except ImportError as e:
            raise ExasolUnavailable("pyexasol is not installed (pip install pyexasol)") from e
        if not (self.cfg["dsn"] and self.cfg["user"] and self.cfg["password"]):
            raise ExasolUnavailable("EXASOL_DSN / EXASOL_USER / EXASOL_PASSWORD are not set")
        kw = dict(dsn=self.cfg["dsn"], user=self.cfg["user"], password=self.cfg["password"], encryption=True)
        if not self.cfg["validate_cert"]:  # Personal editions commonly use self-signed certificates
            kw["websocket_sslopt"] = {"cert_reqs": ssl.CERT_NONE}
        try:
            self.conn = pyexasol.connect(**kw)
        except Exception as e:
            raise ExasolUnavailable(f"Could not connect to Exasol: {e}") from e
        return self

    def health_check(self) -> dict:
        try:
            if self.conn is None: self.connect()
            v = self.conn.execute("SELECT 1 FROM DUAL").fetchval()
            return dict(ok=(v == 1), detail="Connected; SELECT 1 FROM DUAL succeeded")
        except ExasolUnavailable as e:
            return dict(ok=False, detail=str(e))
        except Exception as e:
            return dict(ok=False, detail=f"Query failed: {e}")

    def init_schema(self):
        for stmt in _read("schema.sql").split(";"):
            if stmt.strip():
                self.conn.execute(stmt, {"schema": self.cfg["schema"]})

    def load_dataset(self, d: dict):
        """Re-runnable: recreates tables, then bulk-imports. Sales are densified so zero-sales days exist as rows."""
        import pandas as pd
        from src.inventory.engine import to_wide
        self.init_schema()
        w = to_wide(d["sales"]).stack().rename("units_sold").reset_index()
        w.columns = ["sale_date", "sku", "units_sold"]
        w["sale_date"] = w["sale_date"].dt.date
        prod = d["inventory"].rename(columns={"on_hand": "on_hand"})[["sku", "product_name", "category", "supplier_id", "unit_price", "unit_cost", "moq", "pack_size", "on_hand"]]
        po = d["purchase_orders"].copy()
        for c in ("order_date", "expected_date", "received_date"):
            po[c] = pd.to_datetime(po[c]).dt.date.where(po[c].notna(), None)
        for name, df in (("PRODUCTS", prod), ("SUPPLIERS", d["suppliers"]), ("SALES", w),
                         ("PURCHASE_ORDERS", po[["po_id", "sku", "order_date", "expected_date", "received_date", "quantity"]])):
            self.conn.import_from_pandas(df, (self.cfg["schema"], name))

    def sql_features(self, asof: str):
        """Run features.sql in Exasol; returns a pandas DataFrame."""
        self.conn.execute(f'OPEN SCHEMA "{self.cfg["schema"]}"')
        return self.conn.export_to_pandas(_read("features.sql"), {"asof": asof})

    def rolling_demand(self, sku: str, asof: str):
        self.conn.execute(f'OPEN SCHEMA "{self.cfg["schema"]}"')
        return self.conn.export_to_pandas(_read("rolling_demand.sql"), {"sku": sku, "asof": asof})

    def close(self):
        if self.conn is not None:
            self.conn.close(); self.conn = None
