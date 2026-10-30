-- 7-day and 28-day rolling average daily demand for one SKU (window functions run in Exasol).
SELECT sale_date, units_sold,
  AVG(units_sold) OVER (ORDER BY sale_date ROWS BETWEEN 6  PRECEDING AND CURRENT ROW) AS roll_7,
  AVG(units_sold) OVER (ORDER BY sale_date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS roll_28
FROM sales WHERE sku = {sku} AND sale_date <= TO_DATE({asof}, 'YYYY-MM-DD') ORDER BY sale_date;
