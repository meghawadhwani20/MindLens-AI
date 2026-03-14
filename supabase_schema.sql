create extension if not exists pgcrypto;

create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  full_name text not null,
  email text not null unique,
  password text not null,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.habits (
  id uuid primary key default gen_random_uuid(),
  email text not null references public.users(email) on delete cascade,
  age integer not null,
  sleep_hours double precision not null,
  work_hours double precision not null,
  screen_time double precision not null,
  water_intake double precision not null,
  exercise boolean not null default false,
  meals_per_day integer not null,
  social_interaction text not null,
  caffeine_intake boolean not null default false,
  timestamp timestamptz not null default timezone('utc', now()),
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists habits_email_timestamp_idx
  on public.habits (email, timestamp desc);

create table if not exists public.stress_scans (
  id uuid primary key default gen_random_uuid(),
  email text not null references public.users(email) on delete cascade,
  emotion text,
  emotion_confidence double precision,
  mouth_open boolean,
  eyebrow_raise boolean,
  jaw_clench_score double precision,
  jaw_tension text,
  slouch_score double precision,
  head_tilt_angle double precision,
  shoulder_alignment_diff double precision,
  spine_curve_ratio double precision,
  pose_confidence double precision,
  posture_quality text,
  stress_model_score double precision,
  mindlens_model_score double precision,
  avg_stress double precision,
  overall_average_stress double precision,
  overall_stress text,
  scanned_at timestamptz not null default timezone('utc', now()),
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists stress_scans_email_scanned_at_idx
  on public.stress_scans (email, scanned_at desc);

create table if not exists public.sensor_records (
  id uuid primary key default gen_random_uuid(),
  heart_rate double precision not null,
  spo2 double precision not null,
  temperature double precision not null,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists sensor_records_created_at_idx
  on public.sensor_records (created_at desc);
