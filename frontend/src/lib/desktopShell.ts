import { invoke } from '@tauri-apps/api/core'

/** Commands implemented by the desktop shell (desktop/src-tauri/src/main.rs). */
export const desktopShell = {
  backendExitStatus: () => invoke<string | null>('backend_exit_status'),
  openLogsFolder: () => invoke<void>('open_logs_folder'),
  saveSvg: (contents: string, defaultName: string) => invoke<string | null>('save_svg', { contents, defaultName }),
  modelBKeyStatus: () => invoke<boolean>('model_b_key_status'),
  setModelBKey: (key: string) => invoke<void>('set_model_b_key', { key }),
  clearModelBKey: () => invoke<void>('clear_model_b_key'),
}

export type DesktopShell = typeof desktopShell
