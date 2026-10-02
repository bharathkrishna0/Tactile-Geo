-- TactileGeo durable persistence.
--
-- The FastAPI backend is the only writer and connects with a privileged role,
-- so it bypasses row-level security. RLS is defence in depth: if a browser ever
-- reads these tables directly with a teacher's Supabase JWT, it sees only the
-- projects and sessions that teacher owns. Source images live in private object
-- storage; only their storage key is kept here.

create extension if not exists pgcrypto;

create table if not exists profiles (
    id uuid primary key,
    display_name text,
    created_at timestamptz not null default now()
);

create table if not exists projects (
    id uuid primary key default gen_random_uuid(),
    owner_id uuid references profiles(id) on delete cascade,
    name text not null,
    created_at timestamptz not null default now()
);

-- One conversion of one uploaded diagram. Geometry, QA and the tactile SVG are
-- stored as the current document because they are always read and regenerated
-- as a whole; individual teacher changes are kept in teacher_edits.
create table if not exists sessions (
    id text primary key,
    project_id uuid references projects(id) on delete cascade,
    owner_id uuid references profiles(id) on delete cascade,
    original_filename text not null,
    source_storage_key text,
    source_local_path text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    expires_at timestamptz,
    processing_params jsonb not null default '{}'::jsonb,
    detected_shapes jsonb not null default '[]'::jsonb,
    detected_labels jsonb not null default '[]'::jsonb,
    quality_report jsonb,
    semantic_geometry jsonb,
    simplified_geometry jsonb,
    qa_report jsonb,
    preview_svg text,
    tactile_svg text
);
create index if not exists sessions_expires_at_idx on sessions (expires_at) where expires_at is not null;
create index if not exists sessions_owner_idx on sessions (owner_id);

-- One Model A run (initial processing or regeneration after edits).
create table if not exists processing_runs (
    id bigint generated always as identity primary key,
    session_id text not null references sessions(id) on delete cascade,
    kind text not null,
    params jsonb not null default '{}'::jsonb,
    summary jsonb not null default '{}'::jsonb,
    duration_ms integer,
    created_at timestamptz not null default now()
);
create index if not exists processing_runs_session_idx on processing_runs (session_id, created_at);

create table if not exists teacher_edits (
    id bigint generated always as identity primary key,
    session_id text not null references sessions(id) on delete cascade,
    element_id text not null,
    edit jsonb not null,
    created_at timestamptz not null default now()
);
create index if not exists teacher_edits_session_idx on teacher_edits (session_id, created_at);

create table if not exists audit_events (
    id bigint generated always as identity primary key,
    session_id text references sessions(id) on delete cascade,
    event_type text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);
create index if not exists audit_events_session_idx on audit_events (session_id, created_at);

alter table profiles enable row level security;
alter table projects enable row level security;
alter table sessions enable row level security;
alter table processing_runs enable row level security;
alter table teacher_edits enable row level security;
alter table audit_events enable row level security;

-- Policies reference Supabase's auth.uid(); on a plain Postgres without the
-- Supabase auth schema they are skipped and RLS denies all non-owner roles.
do $$
begin
    if exists (select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
               where n.nspname = 'auth' and p.proname = 'uid') then
        execute 'drop policy if exists profiles_self on profiles';
        execute 'create policy profiles_self on profiles for select using (id = auth.uid())';
        execute 'drop policy if exists projects_owner on projects';
        execute 'create policy projects_owner on projects for select using (owner_id = auth.uid())';
        execute 'drop policy if exists sessions_owner on sessions';
        execute 'create policy sessions_owner on sessions for select using (
            owner_id = auth.uid()
            or project_id in (select id from projects where owner_id = auth.uid()))';
        execute 'drop policy if exists processing_runs_owner on processing_runs';
        execute 'create policy processing_runs_owner on processing_runs for select using (
            session_id in (select id from sessions))';
        execute 'drop policy if exists teacher_edits_owner on teacher_edits';
        execute 'create policy teacher_edits_owner on teacher_edits for select using (
            session_id in (select id from sessions))';
        execute 'drop policy if exists audit_events_owner on audit_events';
        execute 'create policy audit_events_owner on audit_events for select using (
            session_id in (select id from sessions))';
    end if;
end
$$;
