import { describe, expect, it } from 'vitest'
import { shardFor } from '../data'

/**
 * The shard function must stay byte-identical to `shard_for` in
 * `pipeline/publish/browser.py`. These expectations were produced by that
 * function, so a drift on either side fails here.
 */
describe('detail shard assignment', () => {
  const cases: Array<[string, number]> = [
    ['ycoss:5', 0],
    ['ycoss:11940', 32],
    ['ycoss:1', 60],
    ['ycoss:99999', 4],
    ['fixture:abc', 49],
  ]

  it('matches the Python implementation', () => {
    for (const [id, expected] of cases) {
      expect(shardFor(id, 64)).toBe(expected)
    }
  })

  it('always lands inside the shard range', () => {
    for (let i = 0; i < 500; i += 1) {
      const shard = shardFor(`ycoss:${i}`, 64)
      expect(shard).toBeGreaterThanOrEqual(0)
      expect(shard).toBeLessThan(64)
    }
  })

  it('is stable across calls', () => {
    expect(shardFor('ycoss:42', 64)).toBe(shardFor('ycoss:42', 64))
  })
})
