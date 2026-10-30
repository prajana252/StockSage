"""CSV schema validation. Returns (clean_df, errors, warnings); errors must block use of the file."""
import pandas as pd

SCHEMAS = {
    "sales": dict(req=["sale_date", "sku", "units_sold"], dates=["sale_date"], nonneg=["units_sold"], text=["sku"]),
    "inventory": dict(req=["sku", "product_name", "category", "on_hand", "unit_price", "supplier_id"],
                      dates=[], nonneg=["on_hand", "unit_price", "unit_cost", "moq", "pack_size"], text=["sku", "supplier_id"]),
    "suppliers": dict(req=["supplier_id", "supplier_name", "lead_time_days"], dates=[], nonneg=["lead_time_days"], text=["supplier_id"]),
    "purchase_orders": dict(req=["po_id", "sku", "order_date", "expected_date", "quantity"],
                            dates=["order_date", "expected_date", "received_date"], nonneg=["quantity"], text=["sku"]),
}
KEYS = {"inventory": ["sku"], "suppliers": ["supplier_id"], "purchase_orders": ["po_id"], "sales": ["sale_date", "sku"]}


def validate(name: str, df: pd.DataFrame):
    sc, errors, warnings = SCHEMAS[name], [], []
    df = df.copy(); df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in sc["req"] if c not in df.columns]
    if missing:
        return df, [f"{name}: missing required column(s): {', '.join(missing)}"], warnings
    if df.empty:
        return df, [f"{name}: file has no rows"], warnings
    for c in sc["text"]:
        df[c] = df[c].astype(str).str.strip()
    for c in sc["dates"]:
        if c not in df.columns: continue
        raw = df[c]; df[c] = pd.to_datetime(raw, errors="coerce")
        blank = raw.isna() | (raw.astype(str).str.strip() == "")
        bad = int((df[c].isna() & ~blank).sum())
        if bad: errors.append(f"{name}.{c}: {bad} unparseable date(s) (use YYYY-MM-DD)")
        if c != "received_date" and int(blank.sum()): errors.append(f"{name}.{c}: {int(blank.sum())} missing date(s)")
    for c in sc["nonneg"]:
        if c not in df.columns: continue
        raw = df[c]; df[c] = pd.to_numeric(raw, errors="coerce")
        if int((df[c].isna() & raw.notna()).sum()): errors.append(f"{name}.{c}: non-numeric value(s)")
        if c in sc["req"] and df[c].isna().any(): errors.append(f"{name}.{c}: missing value(s)")
        if int((df[c] < 0).sum()): errors.append(f"{name}.{c}: {int((df[c] < 0).sum())} negative value(s) not allowed")
    if df.duplicated(KEYS[name]).any(): errors.append(f"{name}: duplicate rows on key {KEYS[name]}")
    if name == "purchase_orders":
        if "received_date" not in df.columns: df["received_date"] = pd.NaT
        if not errors and (df["expected_date"] < df["order_date"]).any():
            errors.append("purchase_orders: expected_date earlier than order_date")
    if name == "inventory" and not errors:
        for c, default in (("unit_cost", None), ("moq", 1), ("pack_size", 1)):
            fill = df["unit_price"] * 0.65 if default is None else default
            if c not in df.columns:
                warnings.append(f"inventory.{c} not provided: filled with documented default"); df[c] = fill
            else:
                df[c] = df[c].fillna(fill)
        df["moq"] = df["moq"].clip(lower=1); df["pack_size"] = df["pack_size"].clip(lower=1)
    return df, errors, warnings


def cross_check(d: dict):
    errs, warns = [], []
    skus = set(d["inventory"]["sku"])
    extra = set(d["sales"]["sku"]) - skus
    if extra: warns.append(f"{len(extra)} SKU(s) in sales are not in inventory and will be ignored")
    miss = set(d["inventory"]["supplier_id"]) - set(d["suppliers"]["supplier_id"])
    if miss: errs.append(f"inventory references unknown supplier_id(s): {sorted(miss)}")
    return errs, warns
