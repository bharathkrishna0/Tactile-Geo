/**
 * Published Windows installer offered by the browser app's download button.
 *
 * Leave `url` null until `TactileGeo-Setup.exe` is attached to a GitHub Release.
 * After each release, set `url` to that asset's https download link and update
 * `version` and `sizeBytes` from the release page.
 */
export interface DesktopRelease {
  url: string | null
  version: string | null
  sizeBytes: number | null
}

export const DESKTOP_RELEASE: DesktopRelease = {
  url: null,
  version: null,
  sizeBytes: null,
}
