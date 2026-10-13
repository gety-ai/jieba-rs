//! A character-level trie for the dictionary, built for the segmenter's
//! access pattern: from every character of the input, enumerate the
//! dictionary words that start there.
//!
//! Nodes are prefixes of dictionary words. The root's transitions for BMP
//! characters live in a directly indexed table, so the first step of every
//! walk is a cache-resident array load. Every other transition is one entry
//! in an open-addressing hash table keyed by `(parent node, character)`, and
//! the entry carries everything a walk needs to know about the child: the
//! id of the word that ends there and a small Bloom-style filter of the
//! characters its own children start with. One probe per character, no
//! separate lookup to test for a word boundary, and no probe at all for the
//! common case of a character that is not a child, against the several
//! dependent loads per *byte* of a byte-level double-array trie.
//!
//! A node is numbered by where it is stored, the root table's nodes by
//! their character and the hash table's by their slot, so an entry does
//! not spend space on the number of the node it leads to. That makes an
//! entry 12 bytes, and the table is sized to the node count rather than
//! to a power of two, so it can be filled densely. Growing the table moves
//! nodes and with them their numbers, which their children's keys embed,
//! so it re-inserts parents before their children.

/// Width of a word id in an entry's value; the high bits hold the second
/// child filter.
const ID_BITS: u32 = 24;
const ID_MASK: u32 = (1 << ID_BITS) - 1;

/// Word id stored on a node that does not end a dictionary word.
const NO_WORD: u32 = ID_MASK;

/// Widest character the key packing allows: 21 bits, all of Unicode.
const CHAR_BITS: u32 = 21;
const CHAR_MASK: u64 = (1 << CHAR_BITS) - 1;

/// A slot key is `(parent << CHAR_BITS | char) + 1` in the low bits; the
/// 16 bits above hold the first child filter.
const KEY_BITS: u32 = 46;
const KEY_MASK: u64 = (1 << KEY_BITS) - 1;

/// Characters with a direct root transition.
const ROOT_TABLE_LEN: usize = 0x1_0000;

/// Node numbers: the root, then the root table's nodes by character, then
/// the hash table's by slot.
const ROOT: u32 = 0;
const FIRST_SLOT_NODE: u32 = ROOT_TABLE_LEN as u32 + 1;
const MAX_SLOTS: usize = (1 << (KEY_BITS - CHAR_BITS)) - FIRST_SLOT_NODE as usize;

const MULTIPLIER: u64 = 0x9E37_79B9_7F4A_7C15;

/// Load above which the table grows, and the load it is sized for when
/// told how many nodes are coming, in percent. Absent children that pass
/// the filters cost a scan to the end of a cluster, which lengthens fast
/// with the load: segmenting at 75% is as fast as at 60%, at 88% it is
/// 10-15% slower.
const MAX_LOAD_PERCENT: usize = 80;
const RESERVE_LOAD_PERCENT: usize = 75;

/// A hash table entry: the node with key `(parent, char)`.
///
/// Packed to 12 bytes; the fields are only ever copied.
#[derive(Clone, Copy, Default)]
#[repr(C, packed(4))]
struct Slot {
    /// Packed `(parent, char)` plus one, and the first child filter; 0
    /// marks an empty slot.
    key: u64,
    /// Word id and the second child filter.
    value: u32,
}

/// A transition out of the root.
#[derive(Clone, Copy)]
struct RootEntry {
    /// The first child filter in the low bits, and [`PRESENT`].
    filter: u32,
    /// Word id and the second child filter, as in a slot.
    value: u32,
}

/// Marks a root entry that leads to a node.
const PRESENT: u32 = 1 << 31;

/// A node reached by a walk, with what is needed to leave it.
#[derive(Clone, Copy)]
struct Node {
    id: u32,
    /// The first child filter, in the position it has in a slot key.
    key_filter: u64,
    value: u32,
}

impl Node {
    #[inline(always)]
    fn word(self) -> Option<i32> {
        let id = self.value & ID_MASK;
        (id != NO_WORD).then_some(id as i32)
    }
}

/// Where a node's child filters are stored.
#[derive(Clone, Copy)]
enum ParentRef {
    /// The node is the root itself, which has no filters.
    Root,
    RootEntry(usize),
    Slot(usize),
}

#[derive(Clone)]
pub(crate) struct CharTrie {
    root: Vec<RootEntry>,
    slots: Vec<Slot>,
    /// Occupied slots.
    len: usize,
}

#[inline(always)]
fn key(parent: u32, ch: char) -> u64 {
    (((parent as u64) << CHAR_BITS) | ch as u64) + 1
}

/// The parent and character of a (filter-free) key.
#[inline]
fn unpack(key: u64) -> (u32, u64) {
    let k = key - 1;
    ((k >> CHAR_BITS) as u32, k & CHAR_MASK)
}

/// The bits a character sets in a node's two child filters: one of 16 in
/// the key's high bits, one of 8 in the value's high bits. A lookup probes
/// only when both are set; with the fan-out this dictionary has (median 1,
/// p90 3) that rejects most characters that are not children without
/// touching the hash table.
#[inline(always)]
fn filter_bits(ch: char) -> (u64, u32) {
    let h = (ch as u32).wrapping_mul(0x9E37_79B1);
    (1u64 << (KEY_BITS + (h >> 28)), 1u32 << (ID_BITS + ((h >> 25) & 7)))
}

/// Number of slots for `nodes` nodes at `load_percent`.
#[inline]
fn slots_for(nodes: usize, load_percent: usize) -> usize {
    (nodes * 100).div_ceil(load_percent).max(1)
}

impl CharTrie {
    pub(crate) fn new() -> Self {
        Self::with_slots(1 << 10)
    }

    fn with_slots(slots: usize) -> Self {
        CharTrie {
            root: vec![
                RootEntry {
                    filter: 0,
                    value: NO_WORD
                };
                ROOT_TABLE_LEN
            ],
            slots: vec![Slot::default(); slots],
            len: 0,
        }
    }

    /// Make room for `nodes` more nodes outside the root table without
    /// growing (see [`count_slot_nodes`]).
    pub(crate) fn reserve(&mut self, nodes: usize) {
        let needed = slots_for(self.len + nodes, RESERVE_LOAD_PERCENT);
        if needed > self.slots.len() {
            self.rehash(needed);
        }
    }

    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        let hash = key.wrapping_mul(MULTIPLIER);
        ((hash as u128 * self.slots.len() as u128) >> 64) as usize
    }

    /// Index of the slot holding the (filter-free) `key`, or of the empty
    /// slot where it would go.
    #[inline(always)]
    fn probe(&self, key: u64) -> usize {
        let mut i = self.index(key);
        loop {
            let stored = self.slots[i].key;
            if stored & KEY_MASK == key || stored == 0 {
                return i;
            }
            i += 1;
            if i == self.slots.len() {
                i = 0;
            }
        }
    }

    /// The node the root's transition on `ch` leads to.
    #[inline(always)]
    fn root_step(&self, ch: char) -> Option<Node> {
        if (ch as usize) < ROOT_TABLE_LEN {
            let entry = self.root[ch as usize];
            return (entry.filter & PRESENT != 0).then_some(Node {
                id: 1 + ch as u32,
                key_filter: ((entry.filter & !PRESENT) as u64) << KEY_BITS,
                value: entry.value,
            });
        }
        let i = self.probe(key(ROOT, ch));
        self.node_at(i)
    }

    /// The node in slot `i`, if the slot is occupied.
    #[inline(always)]
    fn node_at(&self, i: usize) -> Option<Node> {
        let slot = self.slots[i];
        (slot.key != 0).then_some(Node {
            id: FIRST_SLOT_NODE + i as u32,
            key_filter: slot.key,
            value: slot.value,
        })
    }

    /// The child of `node` on `ch`, if it has one. The filters reject most
    /// absent children before the probe.
    #[inline(always)]
    fn step(&self, node: Node, ch: char) -> Option<Node> {
        let (a, b) = filter_bits(ch);
        if node.key_filter & a == 0 || node.value & b == 0 {
            return None;
        }
        self.node_at(self.probe(key(node.id, ch)))
    }

    /// The id of `word`, if it is in the dictionary.
    pub(crate) fn get(&self, word: &str) -> Option<i32> {
        let mut chars = word.chars();
        let mut node = self.root_step(chars.next()?)?;
        for ch in chars {
            node = self.step(node, ch)?;
        }
        node.word()
    }

    /// Call `emit(char_index, end_index, word_id)` for every dictionary word
    /// starting at every character of `chars` (byte offset and value of
    /// each character of a text), position by position and shortest word
    /// first within a position. `end_index` is the index of the character
    /// after the word.
    pub(crate) fn for_each_prefix_at_every_char(&self, chars: &[(u32, char)], mut emit: impl FnMut(usize, usize, i32)) {
        // Walking a position is a chain of dependent loads that the processor
        // cannot overlap with the next position's chain, so the slot the
        // second character of a position a few steps ahead will need is
        // touched now. By the time the walk arrives it is in cache.
        const LOOKAHEAD: usize = 3;

        let n = chars.len();

        for pos in 0..n {
            if let Some(&[(_, c1), (_, c2)]) = chars.get(pos + LOOKAHEAD..pos + LOOKAHEAD + 2)
                && (c1 as usize) < ROOT_TABLE_LEN
                && self.root[c1 as usize].value & filter_bits(c2).1 != 0
            {
                let i = self.index(key(1 + c1 as u32, c2));
                std::hint::black_box(self.slots[i].key);
            }

            let (_, first) = chars[pos];
            let Some(mut node) = self.root_step(first) else {
                continue;
            };
            let mut next = pos + 1;
            loop {
                if let Some(word_id) = node.word() {
                    emit(pos, next, word_id);
                }
                if next >= n {
                    break;
                }
                let (_, ch) = chars[next];
                let Some(child) = self.step(node, ch) else {
                    break;
                };
                node = child;
                next += 1;
            }
        }
    }

    /// Set the id of `word`, creating its nodes as needed. Returns the id it
    /// had before, if any. An empty word is ignored and returns `None`.
    pub(crate) fn insert(&mut self, word: &str, word_id: i32) -> Option<i32> {
        assert!((0..NO_WORD as i32).contains(&word_id), "word id out of range");
        // Growing renumbers nodes, so make room for the whole word before
        // walking it.
        let new_nodes = word.chars().count();
        if (self.len + new_nodes) * 100 > self.slots.len() * MAX_LOAD_PERCENT {
            self.rehash(slots_for(self.len + new_nodes, MAX_LOAD_PERCENT).max(self.slots.len() * 2));
        }

        let replace = |value: &mut u32| {
            let old = *value & ID_MASK;
            *value = (*value & !ID_MASK) | word_id as u32;
            (old != NO_WORD).then_some(old as i32)
        };
        let mut chars = word.chars().peekable();
        let mut parent = ROOT;
        // Where the filters of `parent` are stored, so they can be updated
        // when `parent` gains a child.
        let mut parent_ref = ParentRef::Root;
        while let Some(ch) = chars.next() {
            let last = chars.peek().is_none();
            if parent == ROOT && (ch as usize) < ROOT_TABLE_LEN {
                let entry = &mut self.root[ch as usize];
                entry.filter |= PRESENT;
                if last {
                    return replace(&mut entry.value);
                }
                parent = 1 + ch as u32;
                parent_ref = ParentRef::RootEntry(ch as usize);
                continue;
            }

            let (a, b) = filter_bits(ch);
            match parent_ref {
                ParentRef::Root => {}
                ParentRef::RootEntry(i) => {
                    self.root[i].filter |= (a >> KEY_BITS) as u32;
                    self.root[i].value |= b;
                }
                ParentRef::Slot(i) => {
                    self.slots[i].key |= a;
                    self.slots[i].value |= b;
                }
            }
            let key = key(parent, ch);
            let i = self.probe(key);
            if self.slots[i].key == 0 {
                self.slots[i] = Slot { key, value: NO_WORD };
                self.len += 1;
            }
            if last {
                let mut value = self.slots[i].value;
                let old = replace(&mut value);
                self.slots[i].value = value;
                return old;
            }
            parent = FIRST_SLOT_NODE + i as u32;
            parent_ref = ParentRef::Slot(i);
        }
        None
    }

    /// Shrink the table to the load it is sized for when told in advance
    /// how many nodes are coming, if growing one node at a time left it
    /// much emptier than that.
    pub(crate) fn shrink_to_fit(&mut self) {
        let fit = slots_for(self.len, RESERVE_LOAD_PERCENT);
        if fit * 3 < self.slots.len() * 2 {
            self.rehash(fit);
        }
    }

    /// Move the nodes to a table of `slots` slots. A node's number changes
    /// with its slot, and its children's keys hold that number, so nodes
    /// are re-inserted depth first, each after its parent.
    fn rehash(&mut self, slots: usize) {
        assert!(slots <= MAX_SLOTS, "dictionary has too many distinct prefixes");
        let old = std::mem::replace(&mut self.slots, vec![Slot::default(); slots]);
        let parent_of = |slot: &Slot| unpack(slot.key & KEY_MASK).0 as usize;

        // Old slots grouped by parent node (a counting sort): the children of
        // node `p` are `by_parent[ends[p - 1]..ends[p]]`.
        let mut ends = vec![0u32; FIRST_SLOT_NODE as usize + old.len()];
        for slot in old.iter().filter(|s| s.key != 0) {
            ends[parent_of(slot)] += 1;
        }
        let mut sum = 0;
        for end in &mut ends {
            sum += *end;
            *end = sum - *end;
        }
        let mut by_parent = vec![0u32; self.len];
        for (i, slot) in old.iter().enumerate().filter(|(_, s)| s.key != 0) {
            let p = parent_of(slot);
            by_parent[ends[p] as usize] = i as u32;
            ends[p] += 1;
        }
        let children = |p: usize| {
            let start = if p == 0 { 0 } else { ends[p - 1] };
            &by_parent[start as usize..ends[p] as usize]
        };

        // The children of the root and of the root table's nodes keep their
        // keys. Every other node is moved after its parent, with the
        // parent's new number in its key.
        let mut stack = Vec::new();
        let fixed = ends[FIRST_SLOT_NODE as usize - 1] as usize;
        for &i in &by_parent[..fixed] {
            let slot = old[i as usize];
            let j = self.probe(slot.key & KEY_MASK);
            self.slots[j] = slot;
            stack.push((i, j as u32));
            while let Some((old_slot, new_slot)) = stack.pop() {
                let new_id = FIRST_SLOT_NODE + new_slot;
                for &i in children(FIRST_SLOT_NODE as usize + old_slot as usize) {
                    let slot = old[i as usize];
                    let (_, ch) = unpack(slot.key & KEY_MASK);
                    let key = (((new_id as u64) << CHAR_BITS) | ch) + 1;
                    let j = self.probe(key);
                    self.slots[j] = Slot {
                        key: key | (slot.key & !KEY_MASK),
                        value: slot.value,
                    };
                    stack.push((i, j as u32));
                }
            }
        }
    }
}

/// Number of nodes outside the root table that inserting `words` into an
/// empty trie creates: exact when the words are sorted, since a word then
/// shares with all earlier words no longer a prefix than with the one
/// just before it, and an overestimate otherwise.
pub(crate) fn count_slot_nodes<'a>(words: impl IntoIterator<Item = &'a str>) -> usize {
    let mut prev = "";
    let mut nodes = 0;
    for word in words {
        let mut common = prev.bytes().zip(word.bytes()).take_while(|(a, b)| a == b).count();
        while !word.is_char_boundary(common) {
            common -= 1;
        }
        nodes += word[common..].chars().count();
        if common == 0 && word.chars().next().is_some_and(|c| (c as usize) < ROOT_TABLE_LEN) {
            nodes -= 1;
        }
        prev = word;
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars_of(text: &str) -> Vec<(u32, char)> {
        text.char_indices().map(|(o, c)| (o as u32, c)).collect()
    }

    /// Dictionary words that are prefixes of `text`, shortest first.
    fn prefixes<'a>(trie: &CharTrie, text: &'a str) -> Vec<(i32, &'a str)> {
        let chars = chars_of(text);
        let mut out = Vec::new();
        trie.for_each_prefix_at_every_char(&chars, |pos, end, id| {
            if pos == 0 {
                let byte_end = chars.get(end).map_or(text.len(), |&(o, _)| o as usize);
                out.push((id, &text[..byte_end]));
            }
        });
        out
    }

    #[test]
    fn test_prefixes_at_every_char() {
        let mut trie = CharTrie::new();
        for (i, w) in ["a", "ab", "abc", "b", "bc", "c", "中", "中国", "国"]
            .iter()
            .enumerate()
        {
            trie.insert(w, i as i32);
        }
        let text = "abc中国x";
        let mut out = Vec::new();
        trie.for_each_prefix_at_every_char(&chars_of(text), |pos, end, id| out.push((pos, end, id)));
        // Position by position, shortest word first; ends are character indices.
        assert_eq!(
            out,
            vec![
                (0, 1, 0),
                (0, 2, 1),
                (0, 3, 2),
                (1, 2, 3),
                (1, 3, 4),
                (2, 3, 5),
                (3, 4, 6),
                (3, 5, 7),
                (4, 5, 8)
            ]
        );
    }

    #[test]
    fn test_insert_get_prefixes() {
        let mut trie = CharTrie::with_slots(4);
        let words = [
            "网",
            "网球",
            "网球拍",
            "球拍",
            "拍卖",
            "拍卖会",
            "a",
            "ab",
            "𦡦",
            "𦡦𦡦",
            "中",
        ];
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.insert(w, i as i32), None);
        }
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32), "{w}");
        }
        assert_eq!(trie.get(""), None);
        assert_eq!(trie.get("网球拍卖"), None);
        assert_eq!(trie.get("球"), None); // prefix node without a word
        assert_eq!(trie.get("x"), None);
        assert_eq!(trie.get("abc"), None);
        assert_eq!(trie.get("𦡦x"), None);

        assert_eq!(
            prefixes(&trie, "网球拍卖会"),
            vec![(0, "网"), (1, "网球"), (2, "网球拍")]
        );
        assert_eq!(prefixes(&trie, "拍卖会def"), vec![(4, "拍卖"), (5, "拍卖会")]);
        assert_eq!(prefixes(&trie, "球"), vec![]);
        assert_eq!(prefixes(&trie, "abc"), vec![(6, "a"), (7, "ab")]);
        assert_eq!(prefixes(&trie, "𦡦𦡦𦡦"), vec![(8, "𦡦"), (9, "𦡦𦡦")]);
        assert_eq!(prefixes(&trie, ""), vec![]);
        assert_eq!(prefixes(&trie, "zzz"), vec![]);
    }

    #[test]
    fn test_reinsert_replaces_id() {
        let mut trie = CharTrie::new();
        assert_eq!(trie.insert("中国", 1), None);
        assert_eq!(trie.insert("中国", 2), Some(1));
        assert_eq!(trie.get("中国"), Some(2));
        assert_eq!(trie.insert("中", 3), None);
        assert_eq!(trie.insert("中", 4), Some(3));
        assert_eq!(prefixes(&trie, "中国人"), vec![(4, "中"), (2, "中国")]);
    }

    /// Many children of one node fill its filters; every child must still be
    /// found and non-children rejected.
    #[test]
    fn test_wide_fanout() {
        let mut trie = CharTrie::with_slots(2);
        let children: Vec<String> = (0x4E00..0x4E00 + 300)
            .map(|c| format!("大{}", char::from_u32(c).unwrap()))
            .collect();
        for (i, w) in children.iter().enumerate() {
            trie.insert(w, i as i32);
        }
        for (i, w) in children.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32));
        }
        assert_eq!(trie.get("大x"), None);
        assert_eq!(trie.get("大"), None);
        assert_eq!(prefixes(&trie, "大a"), vec![]);
        assert_eq!(prefixes(&trie, "大一"), vec![(0, "大一")]);
    }

    /// Growing renumbers the nodes in the hash table, which deep words and
    /// words below non-BMP characters chain through.
    #[test]
    fn test_grows_and_survives_rehash() {
        let mut trie = CharTrie::with_slots(2);
        let words: Vec<String> = (0..5000)
            .map(|i| match i % 3 {
                0 => format!("w{i}x{}", i % 7),
                1 => format!("𦡦{i}深层词语{}", i % 5),
                _ => format!("{i}"),
            })
            .collect();
        for (i, w) in words.iter().enumerate() {
            trie.insert(w, i as i32);
            if i % 997 == 0 {
                for (j, w) in words[..=i].iter().enumerate() {
                    assert_eq!(trie.get(w), Some(j as i32));
                }
            }
        }
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32));
        }
        assert_eq!(prefixes(&trie, "w12x5zzz"), vec![(12, "w12x5")]);
        assert_eq!(prefixes(&trie, "𦡦13深层词语3"), vec![(13, "𦡦13深层词语3")]);
        assert!(trie.slots.len() * MAX_LOAD_PERCENT >= trie.len * 100);
    }

    #[test]
    fn test_reserve_avoids_growth() {
        let words: Vec<String> = (0..10_000).map(|i| format!("{i}")).collect();
        let mut sorted: Vec<&str> = words.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        let nodes = count_slot_nodes(sorted.iter().copied());
        let mut trie = CharTrie::new();
        trie.reserve(nodes);
        let slots = trie.slots.len();
        for (i, w) in words.iter().enumerate() {
            trie.insert(w, i as i32);
        }
        assert_eq!(trie.len, nodes);
        assert_eq!(trie.slots.len(), slots);
        assert_eq!(trie.get("9999"), Some(9999));
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut trie = CharTrie::new();
        let words: Vec<String> = (0..3000).map(|i| format!("词{i}语")).collect();
        for (i, w) in words.iter().enumerate() {
            trie.insert(w, i as i32);
        }
        trie.reserve(100_000);
        trie.shrink_to_fit();
        assert_eq!(trie.slots.len(), slots_for(trie.len, RESERVE_LOAD_PERCENT));
        for (i, w) in words.iter().enumerate() {
            assert_eq!(trie.get(w), Some(i as i32));
        }
        assert_eq!(prefixes(&trie, "词12语3"), vec![(12, "词12语")]);
    }

    #[test]
    fn test_count_slot_nodes() {
        // Root-table characters take no slot; a non-BMP first one does.
        assert_eq!(count_slot_nodes(["a", "ab", "abc", "abd", "b", "中", "中国"]), 4);
        assert_eq!(count_slot_nodes(["𦡦", "𦡦𦡦"]), 2);
        assert_eq!(count_slot_nodes(["中国", "中国", "中央"]), 2);
        // Words sharing a byte but not a character: 一 is E4 B8 80, 丁 E4 B8 81.
        assert_eq!(count_slot_nodes(["一x", "丁y"]), 2);
        assert_eq!(count_slot_nodes([]), 0);
    }
}
