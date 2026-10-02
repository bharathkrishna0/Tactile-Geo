-- Durable Model B jobs and the content-addressed result cache.
--
-- Jobs belong to a session and disappear with it. The cache is keyed by image
-- hash + prompt fingerprint + schema version + model, holds advisory results
-- only (never geometry Model A relies on), and is readable by the backend only.

create table if not exists model_b_jobs (
    id text primary key,
    session_id text not null references sessions(id) on delete cascade,
    status text not null default 'queued'
        check (status in ('queued', 'running', 'completed', 'failed', 'cancelled')),
    created_at timestamptz not null default now(),
    started_at timestamptz,
    completed_at timestamptz,
    attempts integer not null default 0,
    model text not null default '',
    provider text not null default '',
    cache_key text not null default '',
    cache_hit boolean not null default false,
    result jsonb,
    error jsonb
);

create index if not exists model_b_jobs_session_idx on model_b_jobs (session_id, created_at desc);
create index if not exists model_b_jobs_active_idx on model_b_jobs (status)
    where status in ('queued', 'running');

create table if not exists model_b_cache (
    cache_key text primary key,
    result jsonb not null,
    resolved_model text not null default '',
    hits integer not null default 0,
    created_at timestamptz not null default now(),
    last_used_at timestamptz not null default now()
);

create index if not exists model_b_cache_last_used_idx on model_b_cache (last_used_at);

alter table model_b_jobs enable row level security;
-- No policies on the cache: only the service role (the backend) may read it.
alter table model_b_cache enable row level security;

do $$
begin
    if exists (select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
               where n.nspname = 'auth' and p.proname = 'uid') then
        execute 'drop policy if exists model_b_jobs_owner on model_b_jobs';
        execute 'create policy model_b_jobs_owner on model_b_jobs for select using (
            session_id in (select id from sessions))';
    end if;
end
$$;
