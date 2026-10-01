import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

function findStylesheet(): string {
  let dir = process.cwd()
  for (let i = 0; i < 6; i += 1) {
    const candidate = resolve(dir, 'src/styles.css')
    if (existsSync(candidate)) return readFileSync(candidate, 'utf8')
    dir = dirname(dir)
  }
  throw new Error('could not locate src/styles.css from ' + process.cwd())
}

const css = findStylesheet()

function channel(value: number): number {
  const c = value / 255
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

function relativeLuminance(hex: string): number {
  const n = hex.replace('#', '')
  const [r, g, b] = [0, 2, 4].map(i => parseInt(n.slice(i, i + 2), 16))
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)
  const [hi, lo] = la > lb ? [la, lb] : [lb, la]
  return (hi + 0.05) / (lo + 0.05)
}

function token(name: string): string {
  const match = css.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{3,8})`))
  if (!match) throw new Error(`design token --${name} not found in styles.css`)
  return match[1]
}

describe('WCAG AA contrast', () => {
  it('the previously failing #787774 really was below 4.5:1 (documents the regression)', () => {
    expect(contrastRatio('#787774', '#ffffff')).toBeLessThan(4.5)
  })

  it('--color-text-secondary passes AA for body text on white', () => {
    expect(contrastRatio(token('color-text-secondary'), '#ffffff')).toBeGreaterThanOrEqual(4.5)
  })

  it('--color-text-secondary passes AA on the panel background #fff', () => {
    expect(contrastRatio(token('color-text-secondary'), '#ffffff')).toBeGreaterThanOrEqual(4.5)
  })

  it('does not reintroduce #787774 as a text colour anywhere in the stylesheet', () => {
    const declarations = css
      .split('}')
      .map(block => block.trim())
      .filter(block => /color\s*:/.test(block))
      .join('}\n')
    expect(declarations).not.toMatch(/#787774/i)
  })

  it('applies the token to the muted secondary text', () => {
    expect(css).toMatch(/\.muted\s*\{[^}]*color:\s*var\(--color-text-secondary\)/)
  })
})
