"""Reproducible synthetic retail dataset (fixed seed). See docs/DATA_DICTIONARY.md."""
import math
import numpy as np
import pandas as pd

END_DATE = pd.Timestamp("2026-06-30")
N_DAYS = 420
CATEGORIES = {
    "Beverages": ["Sparkling Water 1L", "Cold Brew Coffee", "Orange Juice 1L", "Green Tea Bottles", "Energy Drink 250ml", "Oat Milk 1L"],
    "Dairy & Eggs": ["Whole Milk 1L", "Free-Range Eggs 12", "Greek Yogurt 500g", "Butter 250g", "Cheddar 400g", "Cream 250ml"],
    "Bakery": ["Sourdough Loaf", "Wholemeal Bread", "Bagels 6pk", "Croissants 4pk", "Tortilla Wraps", "Muffins 4pk"],
    "Household": ["Dish Soap 500ml", "Laundry Pods 40", "Paper Towels 6pk", "Toilet Paper 12pk", "Trash Bags 30", "Surface Spray"],
    "Personal Care": ["Shampoo 400ml", "Toothpaste 100ml", "Hand Soap 250ml", "Deodorant", "Face Wipes 25", "Sunscreen SPF50"],
    "Snacks": ["Salted Crisps 150g", "Dark Chocolate 100g", "Mixed Nuts 200g", "Granola Bars 6pk", "Popcorn 90g", "Rice Cakes"],
}
# supplier_id, name, promised lead time (days), mean lateness (days)
SUPPLIERS = [("SUP1", "Northwind Foods", 4, 0.3), ("SUP2", "Harbor Fresh", 3, 0.2),
             ("SUP3", "Metro Wholesale", 7, 1.0), ("SUP4", "Globex Imports", 12, 3.5),
             ("SUP5", "Prime Household", 9, 1.0), ("SUP6", "EverClean Supply", 6, 0.5)]
SCENARIO_IDX = [3, 10, 17, 24, 31]  # demand ramps up late; buyer stops ordering in last 21 days
SCENARIO_SUP = [3, 4, 3, 2, 4]


def generate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=END_DATE, periods=N_DAYS)
    prods, sales_rows, stock_rows, po_rows = [], [], [], []
    po_counter = 1
    flat = [(c, n) for c, names in CATEGORIES.items() for n in names]
    for i, (cat, name) in enumerate(flat):
        sku = f"SKU{i+1:03d}"
        sup = SUPPLIERS[SCENARIO_SUP[SCENARIO_IDX.index(i)]] if i in SCENARIO_IDX else SUPPLIERS[i % len(SUPPLIERS)]
        lead, late = sup[2], sup[3]
        if i in SCENARIO_IDX: kind = "scenario"
        elif i % 6 == 1: kind = "rising"
        elif i % 6 == 2: kind = "seasonal"
        elif i % 6 == 5: kind = "intermittent"
        else: kind = "stable"
        base = rng.uniform(0.4, 1.2) if kind == "intermittent" else rng.uniform(15, 55)
        weekly = rng.uniform(0.85, 1.15, 7); weekly[5:] *= rng.uniform(1.0, 1.3); weekly /= weekly.mean()
        price = round(float(rng.uniform(1.5, 12)), 2)
        pack = int(rng.choice([1, 6, 12])); moq = int(rng.choice([1, 12, 24]))
        prods.append(dict(sku=sku, product_name=name, category=cat, supplier_id=sup[0], unit_price=price,
                          unit_cost=round(price * 0.65, 2), moq=moq, pack_size=pack, demand_profile=kind))
        on_hand = base * (lead + 10); open_pos = []
        for t, d in enumerate(dates):
            mult = 1.0
            if kind == "rising": mult = 1 + 0.004 * max(0, t - (N_DAYS - 150))
            if kind == "seasonal": mult = 1 + 0.35 * math.sin(2 * math.pi * t / 90)
            if kind == "scenario": mult = 1 + 0.03 * max(0, t - (N_DAYS - 28))
            demand = int(rng.poisson(base * weekly[d.dayofweek] * mult))
            on_hand += sum(q for a, q in open_pos if a <= d); open_pos = [p for p in open_pos if p[0] > d]
            sold = int(min(on_hand, demand)); on_hand -= sold
            sales_rows.append((d, sku, sold)); stock_rows.append((d, sku, int(round(on_hand)), demand - sold))
            blocked = kind == "scenario" and t >= N_DAYS - 21
            on_order = sum(q for _, q in open_pos)
            if not blocked and on_hand + on_order <= base * (lead + 7):
                qty = max(moq, math.ceil(base * (lead + 21) / pack) * pack)
                delay = max(0, int(round(rng.normal(late, 1.5))))
                actual = d + pd.Timedelta(days=lead + delay)
                open_pos.append((actual, qty))
                po_rows.append(dict(po_id=f"PO{po_counter:05d}", sku=sku, order_date=d, expected_date=d + pd.Timedelta(days=lead),
                                    received_date=actual if actual <= END_DATE else pd.NaT, quantity=qty))
                po_counter += 1
    sales = pd.DataFrame(sales_rows, columns=["sale_date", "sku", "units_sold"])
    stock_hist = pd.DataFrame(stock_rows, columns=["date", "sku", "on_hand", "unmet_units"])
    products = pd.DataFrame(prods)
    last = stock_hist[stock_hist["date"] == END_DATE].set_index("sku")["on_hand"]
    inventory = products.drop(columns="demand_profile").assign(on_hand=products["sku"].map(last).astype(int))
    inventory = inventory[["sku", "product_name", "category", "on_hand", "unit_price", "unit_cost", "supplier_id", "moq", "pack_size"]]
    suppliers = pd.DataFrame([(s[0], s[1], s[2]) for s in SUPPLIERS], columns=["supplier_id", "supplier_name", "lead_time_days"])
    return dict(sales=sales, inventory=inventory, suppliers=suppliers, purchase_orders=pd.DataFrame(po_rows),
                stock_history=stock_hist, product_meta=products[["sku", "demand_profile"]])


FILES = {"sales": "sample_sales.csv", "inventory": "sample_inventory.csv", "suppliers": "sample_suppliers.csv",
         "purchase_orders": "sample_purchase_orders.csv", "stock_history": "sample_stock_history.csv",
         "product_meta": "sample_product_profiles.csv"}


def write_csvs(out_dir: str = "data", seed: int = 42):
    d = generate(seed)
    for k, f in FILES.items():
        d[k].to_csv(f"{out_dir}/{f}", index=False, date_format="%Y-%m-%d")
    return d
