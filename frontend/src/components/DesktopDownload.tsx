import { DESKTOP_RELEASE, type DesktopRelease } from '../desktopRelease'

function formatSize(bytes: number): string {
  const gb = bytes / 1024 ** 3
  return gb >= 1 ? `about ${gb.toFixed(1)} GB` : `about ${Math.round(bytes / 1024 ** 2)} MB`
}

export default function DesktopDownload({ release = DESKTOP_RELEASE }: { release?: DesktopRelease }) {
  const url = release.url?.startsWith('https://') ? release.url : null
  if (!url) {
    return (
      <div className="desktop-download">
        <button type="button" className="download-button" disabled aria-describedby="desktop-download-note">
          Download Desktop App
        </button>
        <small id="desktop-download-note">The Windows desktop app has not been published yet.</small>
      </div>
    )
  }
  const details = [
    'Windows installer',
    release.version && `v${release.version.replace(/^v/, '')}`,
    release.sizeBytes && formatSize(release.sizeBytes),
  ].filter(Boolean).join(' · ')
  return (
    <div className="desktop-download">
      <a className="button-link download-button" href={url} download aria-describedby="desktop-download-note">
        Download Desktop App
      </a>
      <small id="desktop-download-note">{details}</small>
    </div>
  )
}
