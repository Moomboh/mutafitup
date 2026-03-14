/// FASTA parsing for protein sequences.
///
/// Accepts either FASTA-formatted text (one or more records starting
/// with `>`) or a plain amino-acid sequence. Multi-line sequences are
/// concatenated and whitespace is stripped.
use needletail::parse_fastx_reader;

/// A single parsed sequence with an optional FASTA header.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedSequence {
    /// The FASTA header line (without the leading `>`), or `None` for
    /// plain-text input.
    pub header: Option<String>,
    /// The amino-acid sequence with whitespace removed.
    pub sequence: String,
}

/// Parse user input that may be FASTA or a raw sequence.
///
/// Detection: if any non-blank line starts with `>`, the entire input
/// is treated as FASTA. Otherwise it is treated as a single raw
/// sequence (whitespace stripped).
pub fn parse_input(text: &str) -> Vec<ParsedSequence> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if trimmed.starts_with('>') {
        parse_fasta(trimmed)
    } else {
        vec![ParsedSequence {
            header: None,
            sequence: strip_whitespace(trimmed),
        }]
    }
}

fn parse_fasta(text: &str) -> Vec<ParsedSequence> {
    let mut records = Vec::new();

    let mut reader = match parse_fastx_reader(text.as_bytes()) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    while let Some(result) = reader.next() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // id() returns the full header line after `>` (trimmed of \r)
        let header = std::str::from_utf8(record.id()).ok().map(String::from);

        // seq() returns the sequence with newlines stripped
        let seq = record.seq();
        let sequence = std::str::from_utf8(&seq)
            .map(strip_whitespace)
            .unwrap_or_default();

        if !sequence.is_empty() {
            records.push(ParsedSequence { header, sequence });
        }
    }

    records
}

fn strip_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_sequence() {
        let result = parse_input("MAKLVFGPD");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].header, None);
        assert_eq!(result[0].sequence, "MAKLVFGPD");
    }

    #[test]
    fn test_raw_sequence_with_whitespace() {
        let result = parse_input("  MAKL VFGPD  \n  HELLO  ");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].header, None);
        assert_eq!(result[0].sequence, "MAKLVFGPDHELLO");
    }

    #[test]
    fn test_single_fasta() {
        let input = ">sp|P12345|PROT_HUMAN Some protein\nMAKLVFGPD\n";
        let result = parse_input(input);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].header.as_deref(),
            Some("sp|P12345|PROT_HUMAN Some protein")
        );
        assert_eq!(result[0].sequence, "MAKLVFGPD");
    }

    #[test]
    fn test_multi_line_fasta() {
        let input = ">protein1\nMAKL\nVFGPD\nHELLO\n";
        let result = parse_input(input);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].sequence, "MAKLVFGPDHELLO");
    }

    #[test]
    fn test_multiple_fasta_records() {
        let input = ">prot1\nMAKLVFGPD\n>prot2\nWWWW\n>prot3\nAAAA\n";
        let result = parse_input(input);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].header.as_deref(), Some("prot1"));
        assert_eq!(result[0].sequence, "MAKLVFGPD");
        assert_eq!(result[1].header.as_deref(), Some("prot2"));
        assert_eq!(result[1].sequence, "WWWW");
        assert_eq!(result[2].header.as_deref(), Some("prot3"));
        assert_eq!(result[2].sequence, "AAAA");
    }

    #[test]
    fn test_empty_input() {
        assert!(parse_input("").is_empty());
        assert!(parse_input("   \n  \n  ").is_empty());
    }

    #[test]
    fn test_fasta_without_trailing_newline() {
        let input = ">prot1\nMAKL\n>prot2\nWWWW";
        let result = parse_input(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result[1].sequence, "WWWW");
    }
}
