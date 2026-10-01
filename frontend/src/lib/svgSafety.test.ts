import { describe, expect, it } from 'vitest'
import { sanitizeSvg } from './svgSafety'

describe('sanitizeSvg', () => {
  it('keeps tactile geometry and braille dots', () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="210mm"><g class="stroke"><line x1="0" y1="0" x2="5" y2="5"/></g><circle cx="1" cy="1" r="1"/></svg>'
    const clean = sanitizeSvg(svg)
    expect(clean).toContain('<line')
    expect(clean).toContain('<circle')
    expect(clean).toContain('width="210mm"')
  })

  it('removes scripts, event handlers and external links', () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" onload="alert(1)"><script>alert(2)</script><a xlink:href="javascript:alert(3)"><rect onclick="alert(4)" width="1" height="1"/></a><foreignObject><div/></foreignObject></svg>'
    const clean = sanitizeSvg(svg)
    expect(clean).not.toMatch(/script|onload|onclick|javascript|foreignObject/i)
    expect(clean).toContain('<rect')
  })

  it('rejects input that is not an SVG document', () => {
    expect(sanitizeSvg('<html><body>hi</body></html>')).toBe('')
    expect(sanitizeSvg('not xml <')).toBe('')
    expect(sanitizeSvg('')).toBe('')
  })
})
