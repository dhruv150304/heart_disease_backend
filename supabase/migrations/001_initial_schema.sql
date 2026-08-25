create type public.app_role as enum ('patient', 'doctor', 'clinic_admin');

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  full_name text not null default '',
  role public.app_role not null default 'patient',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.clinician_patients (
  clinician_id uuid not null references public.profiles(id) on delete cascade,
  patient_id uuid not null references public.profiles(id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (clinician_id, patient_id),
  check (clinician_id <> patient_id)
);

create table public.predictions (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null references public.profiles(id) on delete cascade,
  input jsonb not null,
  prediction smallint not null check (prediction in (0, 1)),
  probability smallint not null check (probability between 0 and 100),
  confidence smallint not null check (confidence between 0 and 100),
  risk text not null check (risk in ('Low', 'Medium', 'High')),
  model_version text not null,
  created_at timestamptz not null default now()
);

create index predictions_patient_created_idx on public.predictions(patient_id, created_at desc);

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, full_name, role)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'full_name', ''), 'patient');
  return new;
end;
$$;

create trigger on_auth_user_created after insert on auth.users
for each row execute procedure public.handle_new_user();

alter table public.profiles enable row level security;
alter table public.clinician_patients enable row level security;
alter table public.predictions enable row level security;

create policy "profiles: read own" on public.profiles for select using (auth.uid() = id);
create policy "predictions: patients read own" on public.predictions for select using (auth.uid() = patient_id);
create policy "predictions: clinicians read assigned patients" on public.predictions for select using (
  exists (select 1 from public.clinician_patients cp where cp.clinician_id = auth.uid() and cp.patient_id = predictions.patient_id)
);

-- Do not grant browser clients insert/update/delete policies. The FastAPI service
-- uses the service-role key server-side after validating the user's JWT.
