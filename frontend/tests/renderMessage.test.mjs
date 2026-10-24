import assert from 'node:assert/strict'
import { after, test } from 'node:test'
import { JSDOM } from 'jsdom'
import { marked } from 'marked'

// Use a real DOM with inline handlers enabled to check the rendering boundary.
// No external resources are loaded or network requests made by this fixture.
const dom = new JSDOM('<!doctype html><div id="message"></div>', { runScripts: 'dangerously' })
globalThis.window = dom.window
const { renderMessage } = await import('../src/utils/renderMessage.ts')
const target = dom.window.document.getElementById('message')
const imagePayload = '<img src="missing.png" onerror="window.__xssTriggered = true">'

after(() => {
  delete globalThis.window
  dom.window.close()
})

function dispatchImageError(html) {
  dom.window.__xssTriggered = false
  target.innerHTML = html
  target.querySelector('img').dispatchEvent(new dom.window.Event('error'))
  return dom.window.__xssTriggered
}

test('the previous raw Markdown rendering reproduces an executable image handler', () => {
  assert.equal(dispatchImageError(marked.parse(imagePayload, { async: false })), true)
})

test('rendered messages cannot execute image error handlers', () => {
  assert.equal(dispatchImageError(renderMessage(imagePayload)), false)
  assert.equal(target.querySelector('img').hasAttribute('onerror'), false)
})

test('scriptable HTML and embedded documents are removed', () => {
  for (const payload of [
    '<svg onload="window.__xssTriggered = true"></svg>',
    '<iframe srcdoc="<script>window.__xssTriggered = true</script>"></iframe>',
    '<script>window.__xssTriggered = true</script>',
  ]) {
    target.innerHTML = renderMessage(payload)
    assert.equal(target.querySelector('script, iframe, svg, math'), null)
    for (const element of target.querySelectorAll('*')) {
      assert.equal([...element.attributes].some(attr => /^on/i.test(attr.name)), false)
    }
  }
})

test('unsafe link protocols are removed while link text remains', () => {
  target.innerHTML = renderMessage('<a href="javascript:window.__xssTriggered = true">reference</a>')
  assert.equal(target.querySelector('a').hasAttribute('href'), false)
  assert.equal(target.textContent.trim(), 'reference')
})

test('normal Markdown formatting, tables and safe links remain available', () => {
  target.innerHTML = renderMessage('# Title\n\n**bold** and *emphasis*\n\n- item\n\n`code`\n\n[reference](https://example.com/paper)\n\n| key | value |\n| --- | --- |\n| x | y |')
  assert.equal(target.querySelector('h1').textContent, 'Title')
  assert.equal(target.querySelector('strong').textContent, 'bold')
  assert.equal(target.querySelector('em').textContent, 'emphasis')
  assert.equal(target.querySelector('li').textContent, 'item')
  assert.equal(target.querySelector('code').textContent, 'code')
  assert.equal(target.querySelector('a').getAttribute('href'), 'https://example.com/paper')
  assert.equal(target.querySelectorAll('table tr').length, 2)
})
