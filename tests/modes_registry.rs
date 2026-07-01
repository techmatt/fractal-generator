//! Guard: `specs/modes_registry.json` must stay in lockstep with the `.json` spec
//! files in `specs/`. Asserts (1) the registry's spec keys match exactly the spec
//! files present (no missing, no extra), (2) every tier is valid, and (3) each
//! spec's stamped `"tier"` equals its registry tier. Reuses the hand-rolled
//! `jsonl` scalar readers (the registry is written one flat object per line, so a
//! per-line `field_str` lookup finds exactly that entry's fields).
//!
//! The human-scannable view (`specs/REGISTRY.md`) is generated + also validated by
//! `tools/specs/gen_registry.py`; this test is the enforcement that runs under
//! `cargo test`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use fractal_generator::jsonl;

fn specs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("specs")
}

/// Spec stems present as `specs/*.json` files, excluding the registry itself.
fn spec_files() -> BTreeSet<String> {
    fs::read_dir(specs_dir())
        .expect("read specs/")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".json")?;
            if stem == "modes_registry" {
                None
            } else {
                Some(stem.to_string())
            }
        })
        .collect()
}

/// `spec -> tier` parsed from the registry (one flat object per line).
fn registry_tiers() -> BTreeMap<String, String> {
    let text = fs::read_to_string(specs_dir().join("modes_registry.json")).expect("read registry");
    let mut map = BTreeMap::new();
    for line in text.lines() {
        if let Some(spec) = jsonl::field_str(line, "spec") {
            let tier = jsonl::field_str(line, "tier").unwrap_or_default();
            assert!(
                map.insert(spec.clone(), tier).is_none(),
                "registry: duplicate entry for {spec:?}"
            );
        }
    }
    map
}

#[test]
fn registry_keys_match_spec_files() {
    let files = spec_files();
    let reg: BTreeSet<String> = registry_tiers().keys().cloned().collect();
    let missing: Vec<_> = files.difference(&reg).collect();
    let extra: Vec<_> = reg.difference(&files).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "registry out of sync with specs/: missing from registry={missing:?}, \
         registry keys with no spec file={extra:?}"
    );
}

#[test]
fn every_tier_is_valid() {
    for (spec, tier) in registry_tiers() {
        assert!(
            tier == "promoted" || tier == "niche",
            "{spec:?}: invalid tier {tier:?}"
        );
    }
}

#[test]
fn stamped_tier_matches_registry() {
    for (spec, tier) in registry_tiers() {
        let path = specs_dir().join(format!("{spec}.json"));
        let text = fs::read_to_string(&path).unwrap_or_else(|_| panic!("read {path:?}"));
        let stamped = jsonl::field_str(&text, "tier").unwrap_or_default();
        assert_eq!(
            stamped, tier,
            "{spec:?}: stamped tier {stamped:?} != registry tier {tier:?}"
        );
    }
}
