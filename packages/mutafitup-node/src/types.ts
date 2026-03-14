/**
 * Re-export all types from mutafitup-common.
 *
 * This file exists for backwards compatibility so that relative imports
 * from within this package (e.g. in tests) continue to work.
 */
export type {
  ExportMetadata,
  PreprocessingConfig,
  TaskConfig,
  Predictions,
} from "mutafitup-common";
