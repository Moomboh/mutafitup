/// Sequence preprocessing matching the Python backbone's
/// `preprocess_sequences()` logic.
///
/// Transforms are driven by the `preprocessing` field in
/// `export_metadata.json` so the Rust side stays in sync with
/// the Python export without hard-coding backbone-specific rules.
use super::PreprocessingConfig;

/// Apply preprocessing transforms to raw protein sequences.
///
/// Steps (in order):
/// 1. Character replacements (e.g. rare amino acids O/B/U/Z/J -> X)
/// 2. Space-separate each character (e.g. "MAKLV" -> "M A K L V")
/// 3. Prepend prefix (e.g. "<AA2fold> M A K L V")
pub fn preprocess_sequences(sequences: &[String], config: &PreprocessingConfig) -> Vec<String> {
    sequences
        .iter()
        .map(|seq| preprocess_sequence(seq, config))
        .collect()
}

fn preprocess_sequence(sequence: &str, config: &PreprocessingConfig) -> String {
    let mut result = String::with_capacity(sequence.len() * 2);

    // Character replacements
    if config.char_replacements.is_empty() {
        result.push_str(sequence);
    } else {
        for ch in sequence.chars() {
            let ch_str = ch.to_string();
            if let Some(replacement) = config.char_replacements.get(&ch_str) {
                result.push_str(replacement);
            } else {
                result.push(ch);
            }
        }
    }

    // Space-separate each character
    if config.space_separate {
        let chars: Vec<char> = result.chars().collect();
        result = chars
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" ");
    }

    // Prepend prefix
    if let Some(ref prefix) = config.prefix {
        if config.space_separate {
            result = format!("{prefix} {result}");
        } else {
            result = format!("{prefix}{result}");
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_no_transforms() {
        let config = PreprocessingConfig {
            space_separate: false,
            prefix: None,
            char_replacements: HashMap::new(),
        };
        let result = preprocess_sequences(&["MAKLV".to_string()], &config);
        assert_eq!(result, vec!["MAKLV"]);
    }

    #[test]
    fn test_char_replacements() {
        let mut replacements = HashMap::new();
        replacements.insert("O".to_string(), "X".to_string());
        replacements.insert("B".to_string(), "X".to_string());
        let config = PreprocessingConfig {
            space_separate: false,
            prefix: None,
            char_replacements: replacements,
        };
        let result = preprocess_sequences(&["MOAB".to_string()], &config);
        assert_eq!(result, vec!["MXAX"]);
    }

    #[test]
    fn test_space_separate() {
        let config = PreprocessingConfig {
            space_separate: true,
            prefix: None,
            char_replacements: HashMap::new(),
        };
        let result = preprocess_sequences(&["MET".to_string()], &config);
        assert_eq!(result, vec!["M E T"]);
    }

    #[test]
    fn test_prefix_with_space_separate() {
        let config = PreprocessingConfig {
            space_separate: true,
            prefix: Some("<AA2fold>".to_string()),
            char_replacements: HashMap::new(),
        };
        let result = preprocess_sequences(&["MET".to_string()], &config);
        assert_eq!(result, vec!["<AA2fold> M E T"]);
    }

    #[test]
    fn test_prefix_without_space_separate() {
        let config = PreprocessingConfig {
            space_separate: false,
            prefix: Some("<cls>".to_string()),
            char_replacements: HashMap::new(),
        };
        let result = preprocess_sequences(&["MET".to_string()], &config);
        assert_eq!(result, vec!["<cls>MET"]);
    }
}
