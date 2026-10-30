-- Per-SKU demand, supplier-lateness and open-PO features, computed INSIDE Exasol for an as-of date.
-- Uses only rows dated on/before the as-of date. Assumes the dense daily panel loaded by load_exasol.py (zero-sales days present).
WITH p AS (SELECT TO_DATE({asof}, 'YYYY-MM-DD') AS asof FROM DUAL),
demand AS (
  SELECT s.sku,
    SUM(CASE WHEN s.sale_date > ADD_DAYS(p.asof, -7)  THEN s.units_sold ELSE 0 END) / 7.0  AS avg_7,
    SUM(CASE WHEN s.sale_date > ADD_DAYS(p.asof, -28) THEN s.units_sold ELSE 0 END) / 28.0 AS avg_28,
    STDDEV_POP(CASE WHEN s.sale_date > ADD_DAYS(p.asof, -28) THEN s.units_sold END)        AS std_28
  FROM sales s CROSS JOIN p WHERE s.sale_date <= p.asof GROUP BY s.sku),
late AS (
  SELECT pr.supplier_id, AVG(GREATEST(DAYS_BETWEEN(po.received_date, po.expected_date), 0)) AS mean_lateness, COUNT(*) AS n_received
  FROM purchase_orders po JOIN products pr ON pr.sku = po.sku CROSS JOIN p
  WHERE po.received_date IS NOT NULL AND po.received_date <= p.asof GROUP BY pr.supplier_id),
openpo AS (
  SELECT po.sku, SUM(po.quantity) AS open_qty, MIN(po.expected_date) AS next_expected
  FROM purchase_orders po CROSS JOIN p
  WHERE po.order_date <= p.asof AND (po.received_date IS NULL OR po.received_date > p.asof) GROUP BY po.sku)
SELECT pr.sku, d.avg_7, d.avg_28, d.std_28, pr.on_hand,
       CASE WHEN d.avg_28 > 0 THEN pr.on_hand / d.avg_28 END AS coverage_days,
       su.lead_time_days, COALESCE(l.mean_lateness, 0) AS supplier_lateness, COALESCE(o.open_qty, 0) AS open_po_qty, o.next_expected
FROM products pr JOIN suppliers su ON su.supplier_id = pr.supplier_id
LEFT JOIN demand d ON d.sku = pr.sku LEFT JOIN late l ON l.supplier_id = pr.supplier_id AND l.n_received >= 3
LEFT JOIN openpo o ON o.sku = pr.sku
ORDER BY coverage_days NULLS LAST;
