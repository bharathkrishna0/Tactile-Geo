/**
 * Runtime configuration injected by the TactileGeo desktop shell (desktop/src-tauri)
 * before the page loads. Absent in the browser build, where this is `null`.
 */
export interface DesktopConfig {
  apiBase: string
  logsDir: string
  version: string
  startupError: string | null
}

declare global {
  interface Window {
    __TACTILEGEO_DESKTOP__?: DesktopConfig
  }
}

export function desktopConfig(): DesktopConfig | null {
  return typeof window === 'undefined' ? null : window.__TACTILEGEO_DESKTOP__ ?? null
}
