const BLOCKED_ELEMENTS = ['script', 'foreignObject', 'iframe', 'object', 'embed', 'image', 'use']

/**
 * Re-sanitise backend SVG before it is injected into the page. The backend
 * already sanitises its output; this keeps the browser safe if that ever slips
 * or the frontend is pointed at a different server.
 */
export function sanitizeSvg(svg: string): string {
  if (!svg.trim()) return ''
  const doc = new DOMParser().parseFromString(svg, 'image/svg+xml')
  const root = doc.documentElement
  if (root.nodeName.toLowerCase() !== 'svg' || doc.getElementsByTagName('parsererror').length > 0) return ''
  for (const tag of BLOCKED_ELEMENTS) {
    for (const node of Array.from(doc.getElementsByTagName(tag))) node.remove()
  }
  for (const node of [root, ...Array.from(root.querySelectorAll('*'))]) {
    for (const attribute of Array.from(node.attributes)) {
      const name = attribute.name.toLowerCase()
      const value = attribute.value.replace(/\s+/g, '').toLowerCase()
      if (name.startsWith('on') || ((name === 'href' || name === 'xlink:href') && !value.startsWith('#'))) {
        node.removeAttribute(attribute.name)
      } else if (value.includes('javascript:')) {
        node.removeAttribute(attribute.name)
      }
    }
  }
  return new XMLSerializer().serializeToString(root)
}
