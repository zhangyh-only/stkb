import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveApiBaseUrl } from '../src/core/api-base.ts'

test('defaults to the same-origin API proxy', () => {
  assert.equal(resolveApiBaseUrl(), '/api')
})

test('keeps a complete API base URL unchanged', () => {
  assert.equal(resolveApiBaseUrl('http://localhost:8000/api'), 'http://localhost:8000/api')
})

test('repairs an API origin that omits the project prefix', () => {
  assert.equal(resolveApiBaseUrl('http://localhost:8000/'), 'http://localhost:8000/api')
})
