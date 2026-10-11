import { render, screen } from '@testing-library/react'
import { axe } from 'jest-axe'
import { describe, expect, it } from 'vitest'
import DesktopDownload from './DesktopDownload'

const URL_V1 = 'https://github.com/bharathkrishna0/Tactile-Geo/releases/download/v1.0.0/TactileGeo-Setup.exe'

describe('DesktopDownload', () => {
  it('is disabled with an explanation while no installer is published', async () => {
    const { container } = render(<DesktopDownload release={{ url: null, version: null, sizeBytes: null }} />)
    const button = screen.getByRole('button', { name: /download desktop app/i })
    expect(button).toBeDisabled()
    expect(button).toHaveAccessibleDescription(/windows desktop app has not been published/i)
    expect(screen.queryByRole('link')).toBeNull()
    expect((await axe(container)).violations).toEqual([])
  })

  it('links to the published installer with platform, version and size', async () => {
    const { container } = render(<DesktopDownload release={{ url: URL_V1, version: '1.0.0', sizeBytes: 1.8 * 1024 ** 3 }} />)
    const link = screen.getByRole('link', { name: /download desktop app/i })
    expect(link).toHaveAttribute('href', URL_V1)
    expect(link).toHaveAccessibleDescription('Windows installer · v1.0.0 · about 1.8 GB')
    expect((await axe(container)).violations).toEqual([])
  })

  it('omits version and size when they are unknown', () => {
    render(<DesktopDownload release={{ url: URL_V1, version: null, sizeBytes: null }} />)
    expect(screen.getByRole('link', { name: /download desktop app/i })).toHaveAccessibleDescription('Windows installer')
  })

  it('refuses a non-https location such as a local file', () => {
    render(<DesktopDownload release={{ url: 'file:///C:/Users/dev/TactileGeo-Setup.exe', version: '1.0.0', sizeBytes: 1 }} />)
    expect(screen.getByRole('button', { name: /download desktop app/i })).toBeDisabled()
    expect(screen.queryByRole('link')).toBeNull()
  })
})
