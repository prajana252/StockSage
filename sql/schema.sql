-- Schema name comes from EXASOL_SCHEMA (default STOCKSAGE); the client substitutes it. Re-runnable.
CREATE SCHEMA IF NOT EXISTS {schema!i};
OPEN SCHEMA {schema!i};
CREATE OR REPLACE TABLE products (sku VARCHAR(20), product_name VARCHAR(100), category VARCHAR(50), supplier_id VARCHAR(20),
  unit_price DECIMAL(12,2), unit_cost DECIMAL(12,2), moq DECIMAL(12,0), pack_size DECIMAL(12,0), on_hand DECIMAL(12,0));
CREATE OR REPLACE TABLE suppliers (supplier_id VARCHAR(20), supplier_name VARCHAR(100), lead_time_days DECIMAL(6,0));
CREATE OR REPLACE TABLE sales (sale_date DATE, sku VARCHAR(20), units_sold DECIMAL(12,0));
CREATE OR REPLACE TABLE purchase_orders (po_id VARCHAR(20), sku VARCHAR(20), order_date DATE, expected_date DATE,
  received_date DATE, quantity DECIMAL(12,0));
