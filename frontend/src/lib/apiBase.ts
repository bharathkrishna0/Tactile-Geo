/**
 * Origin of the FastAPI backend.
 *
 * Empty in development, where Vite proxies `/api` to localhost:8000, and when
 * the frontend host rewrites `/api` to the backend. Set `VITE_API_BASE_URL`
 * at build time (e.g. `https://tactilegeo-api.example.org`) when the backend
 * is on another origin; that origin must then be listed in the backend's
 * `CORS_ORIGINS`.
 */
export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
