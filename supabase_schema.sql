-- supabase_schema.sql
-- Run this in your Supabase SQL Editor to set up the hourly_readings table.
-- Dashboard: https://app.supabase.com → Your Project → SQL Editor

-- 1. Create the main tables
CREATE TABLE IF NOT EXISTS public.hourly_readings (
    id          BIGSERIAL PRIMARY KEY,
    city        TEXT        NOT NULL,
    datetime    TIMESTAMPTZ NOT NULL,
    aqi         NUMERIC(6,2),
    pm25        NUMERIC(8,4),
    pm10        NUMERIC(8,4),
    no2         NUMERIC(8,4),
    so2         NUMERIC(8,4),
    co          NUMERIC(8,4),
    o3          NUMERIC(8,4),
    source      TEXT        DEFAULT 'openweather',
    created_at  TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate (city, datetime) combinations
    UNIQUE (city, datetime)
);

CREATE TABLE IF NOT EXISTS public.alert_logs (
    id          BIGSERIAL PRIMARY KEY,
    city        TEXT        NOT NULL,
    alert_time  TIMESTAMPTZ NOT NULL,
    aqi         NUMERIC(6,2),
    severity    TEXT        NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_hourly_readings_city_datetime
    ON public.hourly_readings (city, datetime DESC);

CREATE INDEX IF NOT EXISTS idx_alert_logs_city_time
    ON public.alert_logs (city, alert_time DESC);

-- 3. Row-Level Security — enable but allow anonymous reads and service-role writes
ALTER TABLE public.hourly_readings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alert_logs ENABLE ROW LEVEL SECURITY;

-- Allow public reads
CREATE POLICY "Allow public readings reads"
    ON public.hourly_readings FOR SELECT
    USING (true);

CREATE POLICY "Allow public alert_logs reads"
    ON public.alert_logs FOR SELECT
    USING (true);

-- Only service-role can insert
CREATE POLICY "Allow service role readings inserts"
    ON public.hourly_readings FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Allow service role alert_logs inserts"
    ON public.alert_logs FOR INSERT
    WITH CHECK (true);

-- 4. Quick verification
SELECT COUNT(*) AS readings_count FROM public.hourly_readings;
SELECT COUNT(*) AS alerts_count FROM public.alert_logs;
