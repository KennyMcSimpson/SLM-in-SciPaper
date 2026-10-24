import DOMPurify from 'dompurify'
import { marked } from 'marked'

export function renderMessage(text: string): string {
  const html = marked.parse(text, { async: false })
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } })
}
