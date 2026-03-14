import { describe, it, expect } from "vitest";
import { preprocessSequences } from "../preprocessing.js";
import type { PreprocessingConfig } from "../types.js";

const NO_OP_CONFIG: PreprocessingConfig = {
  space_separate: false,
  prefix: null,
  char_replacements: {},
};

describe("preprocessSequences", () => {
  describe("no transforms", () => {
    it("passes sequences through unchanged", () => {
      const result = preprocessSequences(["MAKLVFG", "PEPTIDE"], NO_OP_CONFIG);
      expect(result).toEqual(["MAKLVFG", "PEPTIDE"]);
    });

    it("handles an empty sequence", () => {
      const result = preprocessSequences([""], NO_OP_CONFIG);
      expect(result).toEqual([""]);
    });

    it("handles a single character", () => {
      const result = preprocessSequences(["M"], NO_OP_CONFIG);
      expect(result).toEqual(["M"]);
    });

    it("handles an empty batch", () => {
      const result = preprocessSequences([], NO_OP_CONFIG);
      expect(result).toEqual([]);
    });
  });

  describe("character replacements only", () => {
    const config: PreprocessingConfig = {
      space_separate: false,
      prefix: null,
      char_replacements: { O: "X", B: "X", U: "X", Z: "X", J: "X" },
    };

    it("replaces mapped characters", () => {
      const result = preprocessSequences(["MOAB"], config);
      expect(result).toEqual(["MXAX"]);
    });

    it("leaves unmapped characters unchanged", () => {
      const result = preprocessSequences(["MAKLVFG"], config);
      expect(result).toEqual(["MAKLVFG"]);
    });

    it("replaces all rare amino acids", () => {
      const result = preprocessSequences(["OBUZJ"], config);
      expect(result).toEqual(["XXXXX"]);
    });

    it("handles mixed mapped and unmapped characters", () => {
      const result = preprocessSequences(["AMOBKUZ"], config);
      expect(result).toEqual(["AMXXKXX"]);
    });
  });

  describe("space separation only", () => {
    const config: PreprocessingConfig = {
      space_separate: true,
      prefix: null,
      char_replacements: {},
    };

    it("space-separates each character", () => {
      const result = preprocessSequences(["MAKLV"], config);
      expect(result).toEqual(["M A K L V"]);
    });

    it("handles a single character", () => {
      const result = preprocessSequences(["M"], config);
      expect(result).toEqual(["M"]);
    });

    it("handles an empty string", () => {
      const result = preprocessSequences([""], config);
      expect(result).toEqual([""]);
    });
  });

  describe("prefix only", () => {
    const config: PreprocessingConfig = {
      space_separate: false,
      prefix: "<CLS>",
      char_replacements: {},
    };

    it("prepends the prefix (no space when not space-separating)", () => {
      const result = preprocessSequences(["MAKLV"], config);
      expect(result).toEqual(["<CLS>MAKLV"]);
    });
  });

  describe("prefix with space separation", () => {
    const config: PreprocessingConfig = {
      space_separate: true,
      prefix: "<AA2fold>",
      char_replacements: {},
    };

    it("prepends prefix with a space before space-separated characters", () => {
      const result = preprocessSequences(["MAKLV"], config);
      expect(result).toEqual(["<AA2fold> M A K L V"]);
    });
  });

  describe("all transforms combined (ProstT5-style)", () => {
    const config: PreprocessingConfig = {
      space_separate: true,
      prefix: "<AA2fold>",
      char_replacements: { O: "X", B: "X", U: "X", Z: "X", J: "X" },
    };

    it("applies replacements, then space-separation, then prefix", () => {
      const result = preprocessSequences(["MOAB"], config);
      // "MOAB" -> "MXAX" (replacements) -> "M X A X" (space-sep) -> "<AA2fold> M X A X" (prefix)
      expect(result).toEqual(["<AA2fold> M X A X"]);
    });

    it("processes a batch of sequences", () => {
      const result = preprocessSequences(["MAKLV", "OBU"], config);
      expect(result).toEqual([
        "<AA2fold> M A K L V",
        "<AA2fold> X X X",
      ]);
    });
  });

  describe("ESMc-style config (all disabled)", () => {
    // Matches the real export_metadata.json from the ESMc 300M model
    const config: PreprocessingConfig = {
      space_separate: false,
      prefix: null,
      char_replacements: {},
    };

    it("passes sequences through unchanged", () => {
      const result = preprocessSequences(["MAKLVFG", "PEPTIDE"], config);
      expect(result).toEqual(["MAKLVFG", "PEPTIDE"]);
    });
  });
});
